@rem build demo.dme
@mkdir .\bin 2>nul

"D:\$BYOND\byond-514.1582\bin\dm.exe"^
 -clean^
 -verbose^
 -max_errors 1^
 .\src\demo.dme

@if not errorlevel 0 goto :eof

@move .\src\demo.dmb .\bin\demo.dmb 1>nul 2>nul
@move .\src\demo.rsc .\bin\demo.rsc 1>nul 2>nul

@rem build lib.c
clang.exe^
 .\src\lib.c^
 -std=c11^
 -Wall^
 -Wextra^
 -Werror^
 -pedantic^
 -fno-builtin^
 -ffreestanding^
 -m32^
 -O2^
 -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include"^
 -fuse-ld=lld-link.exe^
 -Xlinker /nodefaultlib^
 -Xlinker /dll^
 -Xlinker /machine:x86^
 -Xlinker /largeaddressaware^
 -Xlinker /entry:main^
 -Xlinker /libpath:"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\lib\\Win32"^
 -Xlinker cuda.lib^
 -Xlinker kernel32.lib^
 -Xlinker /out:.\bin\life.dll

@if not errorlevel 0 goto :eof

@del .\bin\life.lib 2>nul
@del .\bin\life.exp 2>nul

@rem build cuda kernels
@mkdir .\bin\kernel 2>nul
nvcc.exe^
 .\src\kernel\life.cu^
 -ptx^
 -m32^
 -o .\bin\kernel\life.ptx

@rem broken on windows, requires editing headers
@rem clang++.exe^
@rem   .\src\kernel\life.cu^
@rem   -std=c++14^
@rem   -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH^
@rem   --cuda-path="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6"^
@rem   --cuda-gpu-arch=sm_61
