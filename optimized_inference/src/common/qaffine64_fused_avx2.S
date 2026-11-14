/* x86-64 SysV, AVX2/FMA: one K64 activation row, 192 outputs.
 * (activation, packed, weight sums, scales, bias, output, activation scale, zp)
 * Two passes of twelve 8-column accumulators: twelve independent reductions
 * hide the AVX-VNNI generation's VPDPBUSD latency, where eight cannot.
 * Integer sums are exact in any order. The zero-point correction -zp*sum uses
 * VPMADDWD: |weight sum| <= 64*63 and zp <= 254 fit its signed 16-bit halves
 * (the sum's high half is its sign extension, the multiplier's is zero).
 * Every output keeps the separate FP32 scale product and FMA. */
.intel_syntax noprefix
.text
.p2align 5
.globl dpdf_qaffine64_fused_avx2
#ifdef __ELF__
.hidden dpdf_qaffine64_fused_avx2
.type dpdf_qaffine64_fused_avx2,@function
#endif
dpdf_qaffine64_fused_avx2:
    .cfi_startproc
    vmovd r10d, xmm0
    mov eax, DWORD PTR [rsp+8]
    neg eax
    and eax, 0xffff
    mov r11d, 0x00010001
    vmovd xmm15, eax
    vpbroadcastd ymm15, xmm15
    vmovd xmm14, r11d
    vpbroadcastd ymm14, xmm14
    vpmaddwd ymm0, ymm15, YMMWORD PTR [rdx+0]
    vpmaddwd ymm1, ymm15, YMMWORD PTR [rdx+32]
    vpmaddwd ymm2, ymm15, YMMWORD PTR [rdx+64]
    vpmaddwd ymm3, ymm15, YMMWORD PTR [rdx+96]
    vpmaddwd ymm4, ymm15, YMMWORD PTR [rdx+128]
    vpmaddwd ymm5, ymm15, YMMWORD PTR [rdx+160]
    vpmaddwd ymm6, ymm15, YMMWORD PTR [rdx+192]
    vpmaddwd ymm7, ymm15, YMMWORD PTR [rdx+224]
    vpmaddwd ymm8, ymm15, YMMWORD PTR [rdx+256]
    vpmaddwd ymm9, ymm15, YMMWORD PTR [rdx+288]
    vpmaddwd ymm10, ymm15, YMMWORD PTR [rdx+320]
    vpmaddwd ymm11, ymm15, YMMWORD PTR [rdx+352]
    vpbroadcastd ymm12, DWORD PTR [rdi+0]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+0]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+512]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1024]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1536]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2048]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2560]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3072]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3584]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4096]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4608]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5120]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5632]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+4]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+32]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+544]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1056]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1568]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2080]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2592]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3104]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3616]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4128]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4640]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5152]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5664]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+8]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+64]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+576]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1088]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1600]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2112]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2624]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3136]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3648]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4160]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4672]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5184]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5696]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+12]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+96]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+608]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1120]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1632]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2144]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2656]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3168]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3680]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4192]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4704]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5216]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5728]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+16]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+128]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+640]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1152]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1664]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2176]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2688]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3200]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3712]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4224]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4736]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5248]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5760]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+20]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+160]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+672]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1184]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1696]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2208]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2720]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3232]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3744]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4256]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4768]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5280]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5792]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+24]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+192]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+704]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1216]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1728]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2240]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2752]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3264]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3776]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4288]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4800]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5312]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5824]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+28]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+224]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+736]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1248]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1760]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2272]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2784]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3296]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3808]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4320]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4832]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5344]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5856]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+32]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+256]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+768]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1280]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1792]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2304]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2816]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3328]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3840]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4352]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4864]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5376]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5888]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+36]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+288]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+800]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1312]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1824]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2336]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2848]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3360]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3872]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4384]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4896]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5408]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5920]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+40]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+320]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+832]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1344]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1856]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2368]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2880]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3392]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3904]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4416]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4928]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5440]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5952]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+44]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+352]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+864]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1376]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1888]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2400]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2912]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3424]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3936]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4448]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4960]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5472]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5984]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+48]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+384]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+896]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1408]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1920]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2432]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2944]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3456]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3968]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4480]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4992]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5504]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6016]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+52]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+416]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+928]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1440]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1952]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2464]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2976]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3488]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4000]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4512]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5024]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5536]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6048]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+56]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+448]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+960]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1472]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1984]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2496]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3008]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3520]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4032]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4544]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5056]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5568]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6080]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+60]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+480]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+992]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+1504]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2016]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+2528]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3040]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+3552]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4064]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+4576]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5088]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+5600]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6112]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vmovd xmm15, r10d
    vbroadcastss ymm15, xmm15
    vcvtdq2ps ymm0, ymm0
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+0]
    vfmadd213ps ymm0, ymm13, YMMWORD PTR [r8+0]
    vmovups YMMWORD PTR [r9+0], ymm0
    vcvtdq2ps ymm1, ymm1
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+32]
    vfmadd213ps ymm1, ymm13, YMMWORD PTR [r8+32]
    vmovups YMMWORD PTR [r9+32], ymm1
    vcvtdq2ps ymm2, ymm2
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+64]
    vfmadd213ps ymm2, ymm13, YMMWORD PTR [r8+64]
    vmovups YMMWORD PTR [r9+64], ymm2
    vcvtdq2ps ymm3, ymm3
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+96]
    vfmadd213ps ymm3, ymm13, YMMWORD PTR [r8+96]
    vmovups YMMWORD PTR [r9+96], ymm3
    vcvtdq2ps ymm4, ymm4
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+128]
    vfmadd213ps ymm4, ymm13, YMMWORD PTR [r8+128]
    vmovups YMMWORD PTR [r9+128], ymm4
    vcvtdq2ps ymm5, ymm5
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+160]
    vfmadd213ps ymm5, ymm13, YMMWORD PTR [r8+160]
    vmovups YMMWORD PTR [r9+160], ymm5
    vcvtdq2ps ymm6, ymm6
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+192]
    vfmadd213ps ymm6, ymm13, YMMWORD PTR [r8+192]
    vmovups YMMWORD PTR [r9+192], ymm6
    vcvtdq2ps ymm7, ymm7
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+224]
    vfmadd213ps ymm7, ymm13, YMMWORD PTR [r8+224]
    vmovups YMMWORD PTR [r9+224], ymm7
    vcvtdq2ps ymm8, ymm8
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+256]
    vfmadd213ps ymm8, ymm13, YMMWORD PTR [r8+256]
    vmovups YMMWORD PTR [r9+256], ymm8
    vcvtdq2ps ymm9, ymm9
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+288]
    vfmadd213ps ymm9, ymm13, YMMWORD PTR [r8+288]
    vmovups YMMWORD PTR [r9+288], ymm9
    vcvtdq2ps ymm10, ymm10
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+320]
    vfmadd213ps ymm10, ymm13, YMMWORD PTR [r8+320]
    vmovups YMMWORD PTR [r9+320], ymm10
    vcvtdq2ps ymm11, ymm11
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+352]
    vfmadd213ps ymm11, ymm13, YMMWORD PTR [r8+352]
    vmovups YMMWORD PTR [r9+352], ymm11
    vmovd xmm15, eax
    vpbroadcastd ymm15, xmm15
    vmovd xmm14, r11d
    vpbroadcastd ymm14, xmm14
    vpmaddwd ymm0, ymm15, YMMWORD PTR [rdx+384]
    vpmaddwd ymm1, ymm15, YMMWORD PTR [rdx+416]
    vpmaddwd ymm2, ymm15, YMMWORD PTR [rdx+448]
    vpmaddwd ymm3, ymm15, YMMWORD PTR [rdx+480]
    vpmaddwd ymm4, ymm15, YMMWORD PTR [rdx+512]
    vpmaddwd ymm5, ymm15, YMMWORD PTR [rdx+544]
    vpmaddwd ymm6, ymm15, YMMWORD PTR [rdx+576]
    vpmaddwd ymm7, ymm15, YMMWORD PTR [rdx+608]
    vpmaddwd ymm8, ymm15, YMMWORD PTR [rdx+640]
    vpmaddwd ymm9, ymm15, YMMWORD PTR [rdx+672]
    vpmaddwd ymm10, ymm15, YMMWORD PTR [rdx+704]
    vpmaddwd ymm11, ymm15, YMMWORD PTR [rdx+736]
    vpbroadcastd ymm12, DWORD PTR [rdi+0]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6144]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6656]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7168]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7680]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8192]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8704]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9216]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9728]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10240]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10752]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11264]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11776]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+4]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6176]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6688]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7200]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7712]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8224]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8736]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9248]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9760]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10272]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10784]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11296]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11808]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+8]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6208]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6720]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7232]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7744]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8256]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8768]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9280]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9792]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10304]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10816]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11328]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11840]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+12]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6240]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6752]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7264]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7776]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8288]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8800]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9312]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9824]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10336]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10848]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11360]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11872]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+16]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6272]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6784]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7296]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7808]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8320]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8832]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9344]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9856]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10368]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10880]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11392]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11904]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+20]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6304]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6816]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7328]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7840]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8352]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8864]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9376]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9888]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10400]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10912]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11424]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11936]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+24]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6336]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6848]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7360]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7872]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8384]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8896]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9408]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9920]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10432]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10944]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11456]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11968]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+28]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6368]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6880]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7392]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7904]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8416]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8928]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9440]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9952]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10464]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10976]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11488]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12000]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+32]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6400]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6912]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7424]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7936]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8448]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8960]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9472]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9984]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10496]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11008]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11520]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12032]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+36]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6432]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6944]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7456]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7968]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8480]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8992]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9504]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10016]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10528]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11040]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11552]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12064]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+40]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6464]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6976]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7488]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8000]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8512]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9024]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9536]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10048]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10560]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11072]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11584]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12096]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+44]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6496]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7008]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7520]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8032]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8544]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9056]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9568]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10080]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10592]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11104]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11616]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12128]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+48]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6528]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7040]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7552]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8064]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8576]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9088]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9600]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10112]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10624]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11136]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11648]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12160]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+52]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6560]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7072]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7584]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8096]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8608]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9120]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9632]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10144]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10656]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11168]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11680]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12192]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+56]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6592]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7104]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7616]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8128]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8640]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9152]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9664]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10176]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10688]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11200]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11712]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12224]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vpbroadcastd ymm12, DWORD PTR [rdi+60]
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+6624]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm0, ymm0, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7136]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm1, ymm1, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+7648]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm2, ymm2, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8160]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm3, ymm3, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+8672]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm4, ymm4, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9184]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm5, ymm5, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+9696]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm6, ymm6, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10208]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm7, ymm7, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+10720]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm8, ymm8, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11232]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm9, ymm9, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+11744]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm10, ymm10, ymm13
    vpmaddubsw ymm13, ymm12, YMMWORD PTR [rsi+12256]
    vpmaddwd ymm13, ymm13, ymm14
    vpaddd ymm11, ymm11, ymm13
    vmovd xmm15, r10d
    vbroadcastss ymm15, xmm15
    vcvtdq2ps ymm0, ymm0
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+384]
    vfmadd213ps ymm0, ymm13, YMMWORD PTR [r8+384]
    vmovups YMMWORD PTR [r9+384], ymm0
    vcvtdq2ps ymm1, ymm1
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+416]
    vfmadd213ps ymm1, ymm13, YMMWORD PTR [r8+416]
    vmovups YMMWORD PTR [r9+416], ymm1
    vcvtdq2ps ymm2, ymm2
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+448]
    vfmadd213ps ymm2, ymm13, YMMWORD PTR [r8+448]
    vmovups YMMWORD PTR [r9+448], ymm2
    vcvtdq2ps ymm3, ymm3
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+480]
    vfmadd213ps ymm3, ymm13, YMMWORD PTR [r8+480]
    vmovups YMMWORD PTR [r9+480], ymm3
    vcvtdq2ps ymm4, ymm4
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+512]
    vfmadd213ps ymm4, ymm13, YMMWORD PTR [r8+512]
    vmovups YMMWORD PTR [r9+512], ymm4
    vcvtdq2ps ymm5, ymm5
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+544]
    vfmadd213ps ymm5, ymm13, YMMWORD PTR [r8+544]
    vmovups YMMWORD PTR [r9+544], ymm5
    vcvtdq2ps ymm6, ymm6
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+576]
    vfmadd213ps ymm6, ymm13, YMMWORD PTR [r8+576]
    vmovups YMMWORD PTR [r9+576], ymm6
    vcvtdq2ps ymm7, ymm7
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+608]
    vfmadd213ps ymm7, ymm13, YMMWORD PTR [r8+608]
    vmovups YMMWORD PTR [r9+608], ymm7
    vcvtdq2ps ymm8, ymm8
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+640]
    vfmadd213ps ymm8, ymm13, YMMWORD PTR [r8+640]
    vmovups YMMWORD PTR [r9+640], ymm8
    vcvtdq2ps ymm9, ymm9
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+672]
    vfmadd213ps ymm9, ymm13, YMMWORD PTR [r8+672]
    vmovups YMMWORD PTR [r9+672], ymm9
    vcvtdq2ps ymm10, ymm10
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+704]
    vfmadd213ps ymm10, ymm13, YMMWORD PTR [r8+704]
    vmovups YMMWORD PTR [r9+704], ymm10
    vcvtdq2ps ymm11, ymm11
    vmulps ymm13, ymm15, YMMWORD PTR [rcx+736]
    vfmadd213ps ymm11, ymm13, YMMWORD PTR [r8+736]
    vmovups YMMWORD PTR [r9+736], ymm11
    vzeroupper
    ret
    .cfi_endproc
#ifdef __ELF__
.size dpdf_qaffine64_fused_avx2,.-dpdf_qaffine64_fused_avx2
.section .note.GNU-stack,"",@progbits
#endif
