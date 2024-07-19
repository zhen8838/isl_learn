module {
  func.func @main(%arg0: memref<8x128x384xf32>, %arg1: memref<8x384x512xf32>, %arg2: memref<8x128x512xf32>, %arg3: memref<8x512x64xf32>, %arg4: memref<8x128x64xf32>) {
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 128 {
        affine.for %arg7 = 0 to 512 {
          affine.for %arg8 = 0 to 384 {
            %0 = affine.load %arg0[%arg5, %arg6, %arg8] : memref<8x128x384xf32>
            %1 = affine.load %arg1[%arg5, %arg8, %arg7] : memref<8x384x512xf32>
            %2 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            affine.store %4, %arg2[%arg5, %arg6, %arg7] : memref<8x128x512xf32>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 128 {
        affine.for %arg7 = 0 to 64 {
          affine.for %arg8 = 0 to 512 {
            %0 = affine.load %arg2[%arg5, %arg6, %arg8] : memref<8x128x512xf32>
            %1 = affine.load %arg3[%arg5, %arg8, %arg7] : memref<8x512x64xf32>
            %2 = affine.load %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            affine.store %4, %arg4[%arg5, %arg6, %arg7] : memref<8x128x64xf32>
          }
        }
      }
    }
    return
  }
}

