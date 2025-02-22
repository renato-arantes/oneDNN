/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/dnnl_thread.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

#include "cpu/x64/jit_uni_group_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace data_type;

namespace {
cpu_isa_t get_supported_isa() {
    if (mayiuse(avx512_core)) return avx512_core;
    if (mayiuse(avx2)) return avx2;
    return isa_undef;
}

cpu_isa_t get_io_isa(cpu_isa_t isa, bool has_f16, bool has_bf16) {
    // re-using avx512_core instantiation for xf16
    // re-using avx2 instantiation for xf16
    if (has_f16 || has_bf16)
        return is_superset(isa, avx512_core) ? (has_f16    ? avx512_core_fp16
                               : mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                                           : avx512_core)
                                             : avx2_vnni_2;
    else
        return isa;
}

const bcast_set_t &get_supported_bcast_strategies() {
    // Group norm processes a single group of channels so far. Because of that,
    // the offset per channel must be passed to the kernel but current binary po
    // logic prevents doing it in scalable way. Keeping only `common` for now.
    static const bcast_set_t set_group_norm {broadcasting_strategy_t::scalar};
    return set_group_norm;
}

template <cpu_isa_t isa>
struct kernel_t : public jit_uni_group_normalization_fwd_t::kernel_base_t,
                  public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_group_normalization_fwd_t::kernel_t);

    kernel_t(const group_normalization_pd_t *pd)
        : jit_uni_group_normalization_fwd_t::kernel_base_t(pd)
        , jit_generator(jit_name(), isa)
        , src_d_(pd->src_md())
        , dst_d_(pd->dst_md())
        , C_(pd->C())
        , C_PER_G_(pd->C() / pd->G())
        , simd_w_(vlen / sizeof(float))
        , axis_simd_full_(C_PER_G_ / simd_w_)
        , axis_simd_tail_(C_PER_G_ % simd_w_)
        , use_scale_(pd->use_scale())
        , use_shift_(pd->use_shift())
        , eps_(pd->desc()->group_norm_epsilon) {

        const auto &post_ops = pd->attr()->post_ops_;
        with_postops_ = post_ops.len() != 0;
        with_binary_ = post_ops.find(primitive_kind::binary) != -1;
        with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;

        const auto &attr_scales = pd->attr()->scales_;
        with_src_scales_ = !attr_scales.has_default_values(DNNL_ARG_SRC);
        with_dst_scales_ = !attr_scales.has_default_values(DNNL_ARG_DST);

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx, vmm_tail_mask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx,
                bf16_emu_zmm_2_idx, bf16_emu_zmm_3_idx, reg_tmp,
                bf16_emu_zmm_4_idx);
        io::io_saturation_conf_t io_saturation_conf(
                vmm_zero.getIdx(), vmm_saturation_ubound.getIdx(), reg_tmp);
        const auto io_isa = get_io_isa(isa,
                utils::one_of(f16, src_d_.data_type(), dst_d_.data_type()),
                utils::one_of(bf16, src_d_.data_type(), dst_d_.data_type()));
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, io_isa,
                {src_d_.data_type(), dst_d_.data_type(), f32 /* stats */},
                io_conf, io_tail_conf, io_bf16_conf,
                {{dst_d_.data_type(), io_saturation_conf}});

        VDEBUGINFO(1, primitive, group_normalization,
                "%s:\n    C_=%" PRId64 "\n    C_PER_G_=%" PRId64
                "\n    simd_w_=%zu\n    axis_simd_full_=%" PRId64
                "\n    axis_simd_tail_=%" PRId64
                "\n    use_scale_=%d\n    use_shift_=%d",
                jit_name(), C_, C_PER_G_, simd_w_, axis_simd_full_,
                axis_simd_tail_, use_scale_, use_shift_);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }
    void generate() override {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
        const size_t c_dst_size
                = C_ * types::data_type_size(dst_d_.data_type());

#define PARAM_OFF(x) offsetof(ker_args_t, x)
        if (with_postops_) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = true;
            static const std::size_t tmp_vmm_injector = this->vmm_tmp.getIdx();

            const eltwise_injector::static_params_t esp(true /*save_state*/,
                    reg_po_injector_helper_, elt_inj_opmask, true /*is_fwd*/,
                    false /*use_dst*/);

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    tmp_vmm_injector, this->r14, this->r15, this->r13,
                    preserve_gpr, preserve_vmm,
                    PARAM_OFF(post_ops_binary_rhs_arg_vec), PARAM_OFF(dst),
                    dst_d_, static_cast<size_t>(axis_simd_tail_), tail_opmask,
                    use_exact_tail_scalar_bcast};

            const binary_injector::static_params_t bsp {
                    reg_param, get_supported_bcast_strategies(), rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<isa>>(
                    this, pd_->attr()->post_ops_, bsp, esp);
        }
        preamble();

        io_.init_bf16();
        if (axis_simd_tail_) io_.prepare_tail_mask();

        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_scale, ptr[reg_param + PARAM_OFF(scale)]);
        mov(reg_shift, ptr[reg_param + PARAM_OFF(shift)]);
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
        mov(reg_src_scales, ptr[reg_param + PARAM_OFF(src_scales)]);
        mov(reg_dst_scales, ptr[reg_param + PARAM_OFF(dst_scales)]);
        mov(reg_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
        mov(reg_eps, ptr[reg_param + PARAM_OFF(eps)]);
#undef PARAM_OFF

        // load epsilon
        uni_vmovq(xmm_tmp, reg_eps);
        uni_vbroadcastss(vmm_eps, xmm_tmp);

        // load ones
        mov(reg_tmp, float2int(1.f));
        uni_vmovq(xmm_tmp, reg_tmp);
        uni_vbroadcastss(vmm_ones, xmm_tmp);

        // add block_start to block_size to define block_end
        add(reg_block_end, reg_src);

        Xbyak::Label unroll_loop, end;
        L(unroll_loop);
        {
            cmp(reg_block_end, reg_src);
            jle(end, T_NEAR);

            io_.init_saturate_f32({dst_d_.data_type()});

            // calculate dst
            compute_dst();

            add(reg_src, c_src_size);
            add(reg_dst, c_dst_size);

            jmp(unroll_loop);
        }
        L(end);

        postamble();

        if (with_eltwise_ && postops_injector_)
            postops_injector_->prepare_table(/* generate = */ true);
    }

    void operator()(const void *src, void *dst, const float *scale,
            const float *shift, const float *mean, const float *var,
            const float *src_scales, const float *dst_scales,
            const void *post_ops_binary_rhs_arg_vec,
            const size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.dst = dst;
        args.scale = scale;
        args.shift = shift;
        args.mean = mean;
        args.var = var;
        args.src_scales = src_scales;
        args.dst_scales = dst_scales;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());
        args.eps = eps_;
        args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;

        jit_generator::operator()(&args);
    }

protected:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const Xbyak::AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                             ? yword
                                                        : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    struct ker_args_t {
        const void *src;
        void *dst;
        const float *scale;
        const float *shift;
        const float *mean;
        const float *var;
        const float *src_scales;
        const float *dst_scales;
        const void *post_ops_binary_rhs_arg_vec;
        size_t block_size;
        float eps;
    };

    io::jit_io_multi_dt_helper_t<Vmm> io_;
    const memory_desc_wrapper src_d_, dst_d_;
    const dim_t C_;
    const dim_t C_PER_G_;
    const size_t simd_w_;
    const dim_t axis_simd_full_;
    const dim_t axis_simd_tail_;
    const bool use_scale_ = false;
    const bool use_shift_ = false;
    const float eps_;
    bool with_postops_ = false;
    bool with_binary_ = false;
    bool with_eltwise_ = false;
    bool with_src_scales_ = false;
    bool with_dst_scales_ = false;

    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

    void compute_dst_body(size_t offt_elems, bool tail = false) {
        if (use_scale_) {
            io_[f32]->load(scale_ptr(offt_elems), vmm_scale, tail);
        }
        if (use_shift_) {
            io_[f32]->load(shift_ptr(offt_elems), vmm_shift, tail);
        }
        io_[src_d_.data_type()]->load(src_ptr(offt_elems), vmm_dst, tail);

        // Broadcasting a single mean and var value per group.
        io_[f32]->broadcast(mean_ptr(0), vmm_mean);
        io_[f32]->broadcast(var_ptr(0), vmm_inv_sqrtvar);

        // calculate inv_sqrtvar
        uni_vaddps(vmm_inv_sqrtvar, vmm_inv_sqrtvar, vmm_eps);
        uni_vsqrtps(vmm_inv_sqrtvar, vmm_inv_sqrtvar);
        uni_vdivps(vmm_inv_sqrtvar, vmm_ones, vmm_inv_sqrtvar, vmm_tmp);

        uni_vsubps(vmm_dst, vmm_dst, vmm_mean);
        uni_vmulps(vmm_dst, vmm_dst, vmm_inv_sqrtvar);

        if (use_scale_ && use_shift_)
            uni_vfmadd213ps(vmm_dst, vmm_scale, vmm_shift);
        else {
            if (use_scale_) uni_vmulps(vmm_dst, vmm_dst, vmm_scale);
            if (use_shift_) uni_vaddps(vmm_dst, vmm_dst, vmm_shift);
        }
        if (with_src_scales_) {
            uni_vmovups(vmm_qscale, ptr[reg_src_scales]);
            uni_vmulps(vmm_dst, vmm_dst, vmm_qscale);
        }
        if (with_postops_) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
            if (with_binary_) {
                rhs_arg_params.vmm_idx_to_out_addr.emplace(
                        vmm_dst.getIdx(), dst_ptr());
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_dst.getIdx(), offt_elems * dst_d_.data_type_size());
                if (tail)
                    rhs_arg_params.vmm_tail_idx_.emplace(vmm_dst.getIdx());
            }
            postops_injector_->compute_vector(vmm_dst.getIdx(), rhs_arg_params);
        }
        if (with_dst_scales_) {
            uni_vmovups(vmm_qscale, ptr[reg_dst_scales]);
            uni_vmulps(vmm_dst, vmm_dst, vmm_qscale);
        }
        io_[dst_d_.data_type()]->store(vmm_dst, dst_ptr(offt_elems), tail);
    }

    void compute_dst() {
        for (dim_t i = 0; i < axis_simd_full_; i++)
            compute_dst_body(i * simd_w_);
        if (axis_simd_tail_) compute_dst_body(axis_simd_full_ * simd_w_, true);
    }

    Xbyak::Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + offt * src_d_.data_type_size()];
    }

    Xbyak::Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + offt * dst_d_.data_type_size()];
    }

    Xbyak::Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + offt * sizeof(float)];
    }

    Xbyak::Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + offt * sizeof(float)];
    }

    Xbyak::Address scale_ptr(size_t offt = 0) {
        return vmmword[reg_scale + offt * sizeof(float)];
    }

    Xbyak::Address shift_ptr(size_t offt = 0) {
        return vmmword[reg_shift + offt * sizeof(float)];
    }

    const Xbyak::Reg64 reg_param = abi_param1;
    const Xbyak::Reg64 reg_src = rdx;
    const Xbyak::Reg64 reg_dst = rax;
    const Xbyak::Reg64 reg_mean = rbx;
    const Xbyak::Reg64 reg_scale = r8;
    const Xbyak::Reg64 reg_block_end = r9;
    const Xbyak::Reg64 reg_eps = r10;
    const Xbyak::Reg64 reg_tmp = r11;
    const Xbyak::Reg64 reg_shift = r12;
    const Xbyak::Reg64 reg_var = r13;
    const Xbyak::Reg64 reg_src_scales = r14;
    const Xbyak::Reg64 reg_dst_scales = r15;

    const Vmm vmm_tail_mask = Vmm(0);
    const Vmm vmm_zero = Vmm(5); // In unroll range, safe for dst compute.
    const Vmm vmm_saturation_ubound
            = Vmm(6); // In unroll range, safe for dst compute.
    const Vmm vmm_qscale = Vmm(7);
    const Vmm vmm_scale = Vmm(8); // In unroll range, safe for dst compute.
    const Vmm vmm_shift = Vmm(9); // In unroll range, safe for dst compute.
    const Vmm vmm_ones = Vmm(10);
    const Vmm vmm_eps = Vmm(11);
    const Vmm vmm_mean = Vmm(12);
    const Vmm vmm_inv_sqrtvar = Vmm(13);
    const Vmm vmm_dst = Vmm(14);
    const Vmm vmm_tmp = Vmm(15);
    const Xbyak::Xmm xmm_tmp = Xbyak::Xmm(15);

    const int bf16_emu_zmm_1_idx = 28;
    const int bf16_emu_zmm_2_idx = 29;
    const int bf16_emu_zmm_3_idx = 30;
    const int bf16_emu_zmm_4_idx = 31;
    const int tail_opmask_idx = 1;
    Opmask tail_opmask = Opmask(tail_opmask_idx);

    const int elt_inj_opmask_idx = 2;
    const Xbyak::Reg64 reg_po_injector_helper_ = r14;
    Opmask elt_inj_opmask = Opmask(elt_inj_opmask_idx);
};

template struct kernel_t<avx2>;
template struct kernel_t<avx512_core>;

template <cpu_isa_t isa>
struct kernel_stat_t
    : public jit_uni_group_normalization_fwd_t::kernel_stat_base_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_uni_group_normalization_fwd_t::kernel_stat_t);

    kernel_stat_t(const group_normalization_pd_t *pd, bool compute_var = false)
        : jit_generator(jit_name())
        , src_d_(pd->src_md())
        , compute_var_(compute_var)
        , C_(pd->C())
        , C_PER_G_(C_ / pd->G())
        , SP_(pd->D() * pd->H() * pd->W())
        , simd_w_(vlen / sizeof(float))
        , axis_simd_tail_(C_PER_G_ % simd_w_)
        , c_block_(unroll_c_ * simd_w_)
        , nc_blocks_(C_PER_G_ / c_block_)
        , c_block_tail_((C_PER_G_ % c_block_) - axis_simd_tail_)
        , unroll_c_tail_(c_block_tail_ / simd_w_) {

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx, vmm_tail_mask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx,
                bf16_emu_zmm_2_idx, bf16_emu_zmm_3_idx, reg_tmp,
                bf16_emu_zmm_4_idx);
        const auto io_isa
                = get_io_isa(isa, utils::one_of(f16, src_d_.data_type()),
                        utils::one_of(bf16, src_d_.data_type()));
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, io_isa,
                {src_d_.data_type(), f32 /* stats */}, io_conf, io_tail_conf,
                io_bf16_conf);

        io::io_tail_conf_t io_tail_conf_stats(
                simd_w_, 1, tail_opmask_stats_idx, vmm_tmp.getIdx(), reg_tmp);
        io_stat_ = io::jit_io_multi_dt_helper_t<Vmm>(
                this, io_isa, {f32}, io_conf, io_tail_conf_stats);

        VDEBUGINFO(1, primitive, group_normalization,
                "%s:\n    compute_var_=%d\n    C_=%" PRId64
                "\n    C_PER_G_=%" PRId64
                "\n    simd_w_=%zu\n    axis_simd_tail_=%" PRId64
                "\n    unroll_c_=%" PRId64 "\n    c_block_=%" PRId64
                "\n    nc_blocks_=%" PRId64 "\n    c_block_tail_=%" PRId64
                "\n    unroll_c_tail_=%" PRId64,
                jit_name(), compute_var_, C_, C_PER_G_, simd_w_,
                axis_simd_tail_, unroll_c_, c_block_, nc_blocks_, c_block_tail_,
                unroll_c_tail_);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void generate() override {
        preamble();

        io_.init_bf16();
        if (axis_simd_tail_) io_.prepare_tail_mask();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        if (compute_var_) mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
        mov(reg_src_start, ptr[reg_param + PARAM_OFF(src)]);
#undef PARAM_OFF

        // Initializing registers for unrolling and further reduction of those
        // is called with the maximum unroll value of a `compute_stat_block`
        // function as they operate over vmms, which numeration depends on
        // unroll value.
        const size_t max_unroll = nc_blocks_ ? unroll_c_
                : unroll_c_tail_             ? unroll_c_tail_
                                             : 1;

        if (!compute_var_) {
            for (size_t ur = 0; ur < max_unroll; ur++) {
                uni_vpxor(Vmm_mean(ur), Vmm_mean(ur), Vmm_mean(ur));
            }
        } else {
            for (size_t ur = 0; ur < max_unroll; ur++) {
                uni_vpxor(Vmm_var(ur), Vmm_var(ur), Vmm_var(ur));
            }
        }

        if (nc_blocks_) {
            xor_(reg_nc_block, reg_nc_block);
            Xbyak::Label c_blk_loop, c_blk_loop_end;
            L(c_blk_loop);
            {

                cmp(reg_nc_block, nc_blocks_);
                je(c_blk_loop_end, T_NEAR);

                // calculate mean
                compute_stat_block(unroll_c_);

                add(reg_src_start,
                        c_block_ * types::data_type_size(src_d_.data_type()));
                add(reg_nc_block, 1);

                jmp(c_blk_loop);
            }
            L(c_blk_loop_end);
        }

        if (unroll_c_tail_) {
            compute_stat_block(unroll_c_tail_);
            add(reg_src_start,
                    c_block_tail_ * types::data_type_size(src_d_.data_type()));
        }

        if (axis_simd_tail_) compute_stat_block(1, true);

        // Reduction on registers for Group normalization as the kernel
        // processes a single group at a time.

        // Part 1 is reducing over unrolled registers.
        const Vmm &vmm_stat = !compute_var_ ? vmm_mean : vmm_var;

        Vmm vmm_tmp_max0 = !compute_var_ ? Vmm_mean(0) : Vmm_var(0);
        Vmm vmm_tmp_max1 = !compute_var_ ? Vmm_mean(1) : Vmm_var(1);
        Vmm vmm_tmp_max2 = !compute_var_ ? Vmm_mean(2) : Vmm_var(2);
        Vmm vmm_tmp_max3 = !compute_var_ ? Vmm_mean(3) : Vmm_var(3);

        switch (max_unroll) {
            case 4: {
                uni_vaddps(vmm_tmp_max0, vmm_tmp_max0, vmm_tmp_max1);
                uni_vaddps(vmm_tmp_max2, vmm_tmp_max2, vmm_tmp_max3);
                uni_vaddps(vmm_stat, vmm_tmp_max0, vmm_tmp_max2);
            } break;
            case 3: {
                uni_vaddps(vmm_tmp_max0, vmm_tmp_max0, vmm_tmp_max1);
                uni_vaddps(vmm_stat, vmm_tmp_max0, vmm_tmp_max2);
            } break;
            case 2: {
                uni_vaddps(vmm_stat, vmm_tmp_max0, vmm_tmp_max1);
            } break;
            case 1: {
                uni_vmovups(vmm_stat, vmm_tmp_max0);
            } break;
            default: break;
        }

        // Part 2 is to reduce within a single register.
        reduce_horizontal(vmm_stat, vmm_tmp);

        // Divide a stat by N.
        // Note: the behavior is aligned with with kernel execution model.
        //   Check for `SINGLE_KERNEL_HEURISTIC_ANCHOR` for a pairing spot.
        if (C_PER_G_ >= 32) {
            mov(reg_tmp, float2int(C_PER_G_ * SP_));
            uni_vmovq(xmm_tmp, reg_tmp);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
            uni_vdivps(vmm_stat, vmm_stat, vmm_tmp);
        }

        io_stat_.prepare_tail_mask();
        const auto &stat_addr = !compute_var_ ? mean_ptr(0) : var_ptr(0);
        io_stat_[f32]->store(vmm_stat, stat_addr, true);

        postamble();
    }

    void operator()(
            const void *src, float *mean, size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.mean = mean;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());

        jit_generator::operator()(&args);
    }

    void operator()(const void *src, const float *mean, float *var,
            size_t block_size) const override {
        ker_args_t args;
        args.src = src;
        args.mean = mean;
        args.var = var;
        args.block_size
                = block_size * C_ * types::data_type_size(src_d_.data_type());

        jit_generator::operator()(&args);
    }

protected:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const Xbyak::AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                             ? yword
                                                        : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    struct ker_args_t {
        const void *src;
        const float *mean;
        const float *var;
        size_t block_size;
    };

    const memory_desc_wrapper src_d_;
    const bool compute_var_;
    const dim_t C_;
    const dim_t C_PER_G_;
    const dim_t SP_;
    const size_t simd_w_;
    const dim_t axis_simd_tail_;
    static constexpr dim_t unroll_c_ = 4;
    const dim_t c_block_;
    const dim_t nc_blocks_;
    const dim_t c_block_tail_;
    const dim_t unroll_c_tail_;

    io::jit_io_multi_dt_helper_t<Vmm> io_;
    // `io_stat_` is to store a single element of mean or var.
    io::jit_io_multi_dt_helper_t<Vmm> io_stat_;

    void reduce_horizontal(const Vmm &vstat, const Vmm &vtmp) {
        if (is_superset(isa, avx512_core)) {
            const Zmm &zstat = Zmm(vstat.getIdx());
            const Zmm &ztmp = Zmm(vtmp.getIdx());

            vshuff32x4(ztmp, zstat, zstat, 0x4E); // 256-bit shuffle
            uni_vaddps(vstat, vstat, vtmp);
            vshuff32x4(ztmp, zstat, zstat, 0xB1); // 128/256-bit shuffle
            uni_vaddps(vstat, vstat, vtmp);
        } else if (is_superset(isa, avx2)) {
            const Ymm &ystat = Ymm(vstat.getIdx());
            const Ymm &ytmp = Ymm(vtmp.getIdx());

            vperm2f128(ytmp, ystat, ystat, 0x1); // 128/256-bit shuffle
            uni_vaddps(vstat, vstat, vtmp);
        }
        uni_vshufps(vtmp, vstat, vstat, 0x4E); // 64/128-bit shuffle
        uni_vaddps(vstat, vstat, vtmp);
        uni_vshufps(vtmp, vstat, vstat, 0xB1); // 32/64-bit shuffle
        uni_vaddps(vstat, vstat, vtmp);
    }

    void compute_mean_block(size_t unroll, bool tail = false) {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_sp_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
#undef PARAM_OFF

        mov(reg_src, reg_src_start);
        // add block_start to block_size to define block_end
        add(reg_sp_block_end, reg_src);

        Xbyak::Label sp_blk_loop, sp_blk_loop_end;
        L(sp_blk_loop);
        {
            cmp(reg_sp_block_end, reg_src);
            jle(sp_blk_loop_end, T_NEAR);

            for (size_t ur = 0; ur < unroll; ur++) {
                io_[src_d_.data_type()]->load(
                        src_ptr(ur * simd_w_), Vmm_src(ur), tail);
                uni_vaddps(Vmm_mean(ur), Vmm_mean(ur), Vmm_src(ur));
            }

            add(reg_src, c_src_size);
            jmp(sp_blk_loop);
        }
        L(sp_blk_loop_end);
    }

    void compute_var_block(size_t unroll, bool tail = false) {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_sp_block_end, ptr[reg_param + PARAM_OFF(block_size)]);
#undef PARAM_OFF
        for (size_t ur = 0; ur < unroll; ur++) {
            io_[data_type::f32]->broadcast(mean_ptr(0), Vmm_mean(ur));
        }

        mov(reg_src, reg_src_start);
        // add block_start to block_size to define block_end
        add(reg_sp_block_end, reg_src);

        Xbyak::Label sp_blk_loop, sp_blk_loop_end;
        L(sp_blk_loop);
        {
            cmp(reg_sp_block_end, reg_src);
            jle(sp_blk_loop_end, T_NEAR);

            for (size_t ur = 0; ur < unroll; ur++) {
                io_[src_d_.data_type()]->load(
                        src_ptr(ur * simd_w_), Vmm_src(ur), tail);
            }
            for (size_t ur = 0; ur < unroll; ur++) {
                if (!tail)
                    uni_vsubps(Vmm_src(ur), Vmm_src(ur), Vmm_mean(ur));
                else {
                    // Subtract with mask to keep zeros in spots where there's
                    // no data. Otherwise, subtracting mean and accumulating
                    // towards variance will spoil the right answer.
                    if (is_superset(isa, avx512_core)) {
                        uni_vsubps(Vmm_src(ur) | tail_opmask, Vmm_src(ur),
                                Vmm_mean(ur));
                    } else if (is_superset(isa, avx)) {
                        // Use a scratch zeroed register to keep stats properly
                        // computed.
                        uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
                        uni_vblendvps(Vmm_mean(ur), vmm_tmp, Vmm_mean(ur),
                                vmm_tail_mask);
                        uni_vsubps(Vmm_src(ur), Vmm_src(ur), Vmm_mean(ur));
                    } else {
                        assert(!"unsupported isa");
                    }
                }
            }
            for (size_t ur = 0; ur < unroll; ur++) {
                uni_vfmadd231ps(Vmm_var(ur), Vmm_src(ur), Vmm_src(ur));
            }

            add(reg_src, c_src_size);
            jmp(sp_blk_loop);
        }
        L(sp_blk_loop_end);
    }
    void compute_stat_block(size_t unroll, bool tail = false) {
        if (compute_var_)
            compute_var_block(unroll, tail);
        else
            compute_mean_block(unroll, tail);
    }

    Vmm Vmm_mean(size_t ur = 0) { return Vmm(1 + 0 * unroll_c_ + ur); }
    Vmm Vmm_var(size_t ur = 0) { return Vmm(1 + 1 * unroll_c_ + ur); }
    Vmm Vmm_src(size_t ur = 0) { return Vmm(1 + 2 * unroll_c_ + ur); }

    Xbyak::Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + offt * src_d_.data_type_size()];
    }

    Xbyak::Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + offt * sizeof(float)];
    }

    Xbyak::Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + offt * sizeof(float)];
    }

    const Xbyak::Reg64 reg_param = abi_param1;
    const Xbyak::Reg64 reg_src = rdx;
    const Xbyak::Reg64 reg_src_start = rax;
    const Xbyak::Reg64 reg_mean = rbx;
    const Xbyak::Reg64 reg_sp_block_end = r9;
    const Xbyak::Reg64 reg_nc_block = r10;
    const Xbyak::Reg64 reg_tmp = r11;
    const Xbyak::Reg64 reg_var = r12;

    const Vmm vmm_tail_mask = Vmm(0);
    const Vmm vmm_tmp = Vmm(13);
    const Xmm xmm_tmp = Xmm(13);
    const Vmm vmm_var = Vmm(14);
    const Vmm vmm_mean = Vmm(15);

    const int bf16_emu_zmm_1_idx = 28;
    const int bf16_emu_zmm_2_idx = 29;
    const int bf16_emu_zmm_3_idx = 30;
    const int bf16_emu_zmm_4_idx = 31;
    const int tail_opmask_idx = 1;
    const int tail_opmask_stats_idx = 2;
    Opmask tail_opmask = Opmask(tail_opmask_idx);
};

template struct kernel_stat_t<avx2>;
template struct kernel_stat_t<avx512_core>;

} // namespace

jit_uni_group_normalization_fwd_t::kernel_base_t *
jit_uni_group_normalization_fwd_t::kernel_base_t::create(
        const group_normalization_pd_t *pd) {
    if (mayiuse(avx512_core)) {
        return new kernel_t<avx512_core>(pd);
    } else if (mayiuse(avx2)) {
        return new kernel_t<avx2>(pd);
    } else {
        assert(!"kernel is empty.");
        return nullptr;
    }
}

jit_uni_group_normalization_fwd_t::kernel_stat_base_t *
jit_uni_group_normalization_fwd_t::kernel_stat_base_t::create(
        const group_normalization_pd_t *apd, bool compute_var) {
    if (mayiuse(avx512_core)) {
        return new kernel_stat_t<avx512_core>(apd, compute_var);
    } else if (mayiuse(avx2)) {
        return new kernel_stat_t<avx2>(apd, compute_var);
    } else {
        assert(!"kernel is empty.");
        return nullptr;
    }
}

status_t jit_uni_group_normalization_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace format_tag;
    using skip_mask_t = primitive_attr_t::skip_mask_t;

    const memory_desc_wrapper src_d(src_md());

    VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_GNORM(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_GNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_GNORM(utils::one_of(src_md()->data_type, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(utils::one_of(dst_md()->data_type, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(IMPLICATION(utils::one_of(bf16, src_md()->data_type,
                                        dst_md()->data_type),
                            mayiuse(avx512_core) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_GNORM(IMPLICATION(utils::one_of(f16, src_md()->data_type,
                                        dst_md()->data_type),
                            mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_GNORM(attr()->has_default_values(
                            skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_GNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_GNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_GNORM(
            memory_desc_matches_one_of_tag(*src_md(), ndhwc, nhwc, nwc, nc),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_GNORM(
            memory_desc_matches_one_of_tag(*dst_md(), ndhwc, nhwc, nwc, nc),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    // Instance Normalization is handled in a different implementation. This
    // implementation has some turns in the kernel that is done differently
    // due to processing a group and not having an ability to process full
    // registers of channels.
    // It has also some dispatching logic in parallelization to process groups
    // differently, see the comment in a correspondent section.
    const size_t C_PER_G = C() / G();
    VDISPATCH_GNORM(C_PER_G > 1, "Instance norm is not supported");

    auto post_ops_ok = [&]() -> bool {
        const std::vector<injector::post_op_type> accepted_post_ops
                = {injector::eltwise, injector::binary, injector::sum};
        const memory_desc_wrapper dst_d(dst_md());
        injector::post_ops_ok_args_t post_ops_args(get_supported_isa(),
                accepted_post_ops, attr()->post_ops_, &dst_d, true, true, true,
                true, get_supported_bcast_strategies());

        return injector::post_ops_ok(post_ops_args);
    };
    VDISPATCH_GNORM(attr_.set_default_formats(dst_md(0)) == status::success,
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_GNORM(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);

    nthr_ = dnnl_get_max_threads();
    auto scratchpad = scratchpad_registry().registrar();
    if (!stats_is_src()) {
        using namespace memory_tracking::names;
        // C() is used here for convenience, to let C++ reduce over the group.
        // TODO: replace with G() instead and make reduction in registers.
        const size_t stats_size = MB() * C();
        const size_t stats_reduction_buf_sz = stats_size * nthr_;
        scratchpad.template book<float>(
                key_gnorm_reduction, stats_reduction_buf_sz);
        if (!is_training()) {
            scratchpad.template book<float>(key_gnorm_tmp_mean, stats_size);
            scratchpad.template book<float>(key_gnorm_tmp_var, stats_size);
        }
    }

    return status::success;
}

status_t jit_uni_group_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto stat_reduction = scratchpad.template get<float>(key_gnorm_reduction);
    auto tmp_mean = scratchpad.template get<float>(key_gnorm_tmp_mean);
    auto tmp_var = scratchpad.template get<float>(key_gnorm_tmp_var);

    float *mean {nullptr}, *variance {nullptr};
    mean = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
            : pd()->is_training() ? CTX_OUT_MEM(float *, DNNL_ARG_MEAN)
                                  : tmp_mean;
    variance = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
            : pd()->is_training() ? CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE)
                                  : tmp_var;

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const dim_t N = src_d.dims()[0];
    const dim_t C_padded = src_d.padded_dims()[1];
    const dim_t C = src_d.dims()[1];
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const dim_t G = pd()->G();
    const dim_t C_PER_G = C / G;
    const dim_t SP = D * H * W;

    const bool calculate_stats = !pd()->stats_is_src();
    const int nthr = pd()->nthr_;

    // There are two algorithms to distribute the problem among threads:
    // * Single-threaded-group - it gives each thread a whole group and runs
    //   it through all kernels. In this case there are no dependencies and
    //   no need to sync between threads. Beneficial for a decent number of
    //   channels in a group and short spatial.
    //   Note: this algorithm requires a modification in the kernel that would
    //   divide mean and variance by N. In case the heuristic change, the other
    //   place must be updated accordingly.
    //   Check for `SINGLE_KERNEL_HEURISTIC_ANCHOR` for a pairing spot.
    //
    // * Multi-threaded-group - it gives a single group to several threads.
    //   In this case, synchronization is required, to collect proper mean and
    //   variance values.
    //   Turned out to be faster as, otherwise, threads would fight for memory
    //   which overcomes synchronization price.
    if (C_PER_G >= 32) {
        parallel(nthr, [&](const int ithr, const int nthr) {
            dim_t g_start = 0, g_end = 0;
            balance211(G * N, nthr, ithr, g_start, g_end);
            if (g_start == g_end) return;

            for (dim_t i = g_start; i < g_end; i++) {
                dim_t stride_n = SP * C_padded;
                const size_t data_off = (i / G) * stride_n + (i % G) * C_PER_G;
                const char *__restrict src_ptr = static_cast<const char *>(src)
                        + data_off * src_d.data_type_size();
                char *__restrict dst_ptr = static_cast<char *>(dst)
                        + data_off * dst_d.data_type_size();
                const float *__restrict scale_ptr
                        = scale ? scale + (i % G) * C_PER_G : nullptr;
                const float *__restrict shift_ptr
                        = shift ? shift + (i % G) * C_PER_G : nullptr;
                float *mean_ptr = mean + i;
                float *var_ptr = variance + i;

                if (calculate_stats) {
                    (*kernel_mean_)(src_ptr, mean_ptr, SP);
                    (*kernel_var_)(src_ptr, mean_ptr, var_ptr, SP);
                }
                (*kernel_)(src_ptr, dst_ptr, scale_ptr, shift_ptr, mean_ptr,
                        var_ptr, src_scales, dst_scales,
                        post_ops_binary_rhs_arg_vec.data(), SP);
            }
        });
    } else {
        dim_t nthr_per_g = std::min(static_cast<dim_t>(nthr), G);
        assert(nthr_per_g <= nthr);

        auto reduce = [&](float *stat, const float *tmp_stat) {
            for (dim_t g = 0; g < G * N; ++g)
                stat[g] = 0.f;

            for_(dim_t n = 0; n < N; n++)
            for_(dim_t ithr = 0; ithr < nthr_per_g; ithr++)
            for (dim_t g = 0; g < G; g++) {
                stat[n * G + g] += tmp_stat[n * nthr_per_g * G + ithr * G + g];
            }

            for (dim_t g = 0; g < G * N; ++g)
                stat[g] /= C_PER_G * SP;
        };

        if (calculate_stats) {
            parallel(nthr, [&](const int ithr, const int nthr) {
                dim_t chunk_start = 0, chunk_end = 0;
                balance211(
                        G * N * nthr_per_g, nthr, ithr, chunk_start, chunk_end);
                if (chunk_start == chunk_end) return;

                dim_t g_per_n = G * nthr_per_g;
                dim_t SP_chunk = SP / nthr_per_g;

                for (dim_t i = chunk_start; i < chunk_end; i++) {
                    dim_t ithr_stride_n = (i / g_per_n) * C_padded * SP;
                    dim_t ithr_stride_g = (i % G) * C_PER_G;
                    dim_t ithr_stride_sp
                            = ((i % g_per_n) / G) * C_padded * SP_chunk;
                    const size_t data_off = (size_t)ithr_stride_n
                            + ithr_stride_g + ithr_stride_sp;
                    const char *__restrict src_ptr
                            = static_cast<const char *>(src)
                            + data_off * src_d.data_type_size();

                    float *mean_ptr = stat_reduction + i;

                    dim_t SP_tail_chunk = SP - ((i % g_per_n) / G) * SP_chunk;
                    dim_t kernel_sp_block_size
                            = (((i % g_per_n) / G) == nthr_per_g - 1)
                            ? SP_tail_chunk
                            : SP_chunk;
                    (*kernel_mean_)(src_ptr, mean_ptr, kernel_sp_block_size);
                }
            });
            reduce(mean, stat_reduction);

            parallel(nthr, [&](const int ithr, const int nthr) {
                dim_t chunk_start = 0, chunk_end = 0;
                balance211(
                        G * N * nthr_per_g, nthr, ithr, chunk_start, chunk_end);
                if (chunk_start == chunk_end) return;

                dim_t g_per_n = G * nthr_per_g;
                dim_t SP_chunk = SP / nthr_per_g;

                for (dim_t i = chunk_start; i < chunk_end; i++) {
                    dim_t ithr_stride_n = (i / g_per_n) * C_padded * SP;
                    dim_t ithr_stride_g = (i % G) * C_PER_G;
                    dim_t ithr_stride_sp
                            = ((i % g_per_n) / G) * C_padded * SP_chunk;
                    const size_t data_off = (size_t)ithr_stride_n
                            + ithr_stride_g + ithr_stride_sp;
                    const char *__restrict src_ptr
                            = static_cast<const char *>(src)
                            + data_off * src_d.data_type_size();

                    float *mean_ptr = mean + (i % G) + (i / g_per_n) * G;
                    float *var_ptr = stat_reduction + i;

                    dim_t SP_tail_chunk = SP - ((i % g_per_n) / G) * SP_chunk;
                    dim_t kernel_sp_block_size
                            = (((i % g_per_n) / G) == nthr_per_g - 1)
                            ? SP_tail_chunk
                            : SP_chunk;
                    (*kernel_var_)(
                            src_ptr, mean_ptr, var_ptr, kernel_sp_block_size);
                }
            });
            reduce(variance, stat_reduction);
        }

        parallel(nthr, [&](const int ithr, const int nthr) {
            dim_t chunk_start = 0, chunk_end = 0;
            balance211(G * N * nthr_per_g, nthr, ithr, chunk_start, chunk_end);
            if (chunk_start == chunk_end) return;

            dim_t g_per_n = G * nthr_per_g;
            dim_t SP_chunk = SP / nthr_per_g;

            for (dim_t i = chunk_start; i < chunk_end; i++) {
                dim_t ithr_stride_n = (i / g_per_n) * C_padded * SP;
                dim_t ithr_stride_g = (i % G) * C_PER_G;
                dim_t ithr_stride_sp
                        = ((i % g_per_n) / G) * C_padded * SP_chunk;
                const size_t data_off = (size_t)ithr_stride_n + ithr_stride_g
                        + ithr_stride_sp;
                const char *__restrict src_ptr = static_cast<const char *>(src)
                        + data_off * src_d.data_type_size();
                char *__restrict dst_ptr = static_cast<char *>(dst)
                        + data_off * dst_d.data_type_size();
                const float *__restrict scale_ptr
                        = scale ? scale + (i % G) * C_PER_G : nullptr;
                const float *__restrict shift_ptr
                        = shift ? shift + (i % G) * C_PER_G : nullptr;

                float *mean_ptr = mean + (i % G) + (i / g_per_n) * G;
                float *var_ptr = variance + (i % G) + (i / g_per_n) * G;

                dim_t SP_tail_chunk = SP - ((i % g_per_n) / G) * SP_chunk;
                dim_t kernel_sp_block_size
                        = (((i % g_per_n) / G) == nthr_per_g - 1)
                        ? SP_tail_chunk
                        : SP_chunk;
                (*kernel_)(src_ptr, dst_ptr, scale_ptr, shift_ptr, mean_ptr,
                        var_ptr, src_scales, dst_scales,
                        post_ops_binary_rhs_arg_vec.data(),
                        kernel_sp_block_size);
            }
        });
    }

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
