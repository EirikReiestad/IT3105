; ModuleID = 'probe8.3c9c4e3f-cgu.0'
source_filename = "probe8.3c9c4e3f-cgu.0"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; core::f64::<impl f64>::to_ne_bytes
; Function Attrs: inlinehint uwtable
define internal i64 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$11to_ne_bytes17hfaed9c47c6590fecE"(double %self) unnamed_addr #0 {
start:
  %0 = alloca i64, align 8
  %1 = alloca [8 x i8], align 1
; call core::f64::<impl f64>::to_bits
  %_2 = call i64 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$7to_bits17h1b0fe6b1baf86e03E"(double %self)
  br label %bb1

bb1:                                              ; preds = %start
; call core::num::<impl u64>::to_ne_bytes
  %2 = call i64 @"_ZN4core3num21_$LT$impl$u20$u64$GT$11to_ne_bytes17h07192fa81a324ed2E"(i64 %_2)
  store i64 %2, i64* %0, align 8
  %3 = bitcast [8 x i8]* %1 to i8*
  %4 = bitcast i64* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %3, i8* align 8 %4, i64 8, i1 false)
  br label %bb2

bb2:                                              ; preds = %bb1
  %5 = bitcast [8 x i8]* %1 to i64*
  %6 = load i64, i64* %5, align 1
  ret i64 %6
}

; core::f64::<impl f64>::to_bits
; Function Attrs: inlinehint uwtable
define internal i64 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$7to_bits17h1b0fe6b1baf86e03E"(double %self) unnamed_addr #0 {
start:
  %0 = alloca i64, align 8
  %1 = bitcast double %self to i64
  store i64 %1, i64* %0, align 8
  %2 = load i64, i64* %0, align 8
  br label %bb1

bb1:                                              ; preds = %start
  ret i64 %2
}

; core::num::<impl u64>::to_ne_bytes
; Function Attrs: inlinehint uwtable
define internal i64 @"_ZN4core3num21_$LT$impl$u20$u64$GT$11to_ne_bytes17h07192fa81a324ed2E"(i64 %self) unnamed_addr #0 {
start:
  %0 = alloca [8 x i8], align 1
  %1 = bitcast [8 x i8]* %0 to i64*
  store i64 %self, i64* %1, align 1
  br label %bb1

bb1:                                              ; preds = %start
  %2 = bitcast [8 x i8]* %0 to i64*
  %3 = load i64, i64* %2, align 1
  ret i64 %3
}

; probe8::probe
; Function Attrs: uwtable
define void @_ZN6probe85probe17h477253705e01bbd6E() unnamed_addr #1 {
start:
  %0 = alloca i64, align 8
  %_1 = alloca [8 x i8], align 1
; call core::f64::<impl f64>::to_ne_bytes
  %1 = call i64 @"_ZN4core3f6421_$LT$impl$u20$f64$GT$11to_ne_bytes17hfaed9c47c6590fecE"(double 3.140000e+00)
  store i64 %1, i64* %0, align 8
  %2 = bitcast [8 x i8]* %_1 to i8*
  %3 = bitcast i64* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %2, i8* align 8 %3, i64 8, i1 false)
  br label %bb1

bb1:                                              ; preds = %start
  ret void
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

attributes #0 = { inlinehint uwtable "target-cpu"="x86-64" }
attributes #1 = { uwtable "target-cpu"="x86-64" }
attributes #2 = { argmemonly nofree nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"PIC Level", i32 2}
