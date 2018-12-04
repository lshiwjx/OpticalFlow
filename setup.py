# from setuptools.command.build_ext import build_ext
# import setuptools
# import sys
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
ext_modules = [
    CppExtension(
        'vcap',
        ['video_capture.cpp'],
        include_dirs=[
            '/home/lshi/Application/Anaconda/include',
            "/home/lshi/Application/Anaconda/include/opencv/"
        ],
        library_dirs=['/home/lshi/Application/Anaconda/lib64/'],
        libraries=['opencv_cudacodec', 'opencv_cudawarping']
        # libraries=['opencv_core', 'opencv_cudacodec', 'opencv_highgui', 'opencv_imgproc']
    ),
]
setup(
    name='vcap',
    ext_modules=ext_modules,
    extra_cflags=['-O3'],
    cmdclass={
        'build_ext': BuildExtension
    })


# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

# class get_pybind_include(object):
#     def __init__(self, user=False):
#         self.user = user
#
#     def __str__(self):
#         import pybind11
#         return pybind11.get_include(self.user)

# def has_flag(compiler, flagname):
#     """Return a boolean indicating whether a flag name is supported on
#     the specified compiler.
#     """
#     import tempfile
#     with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
#         f.write('int main (int argc, char **argv) { return 0; }')
#         try:
#             compiler.compile([f.name], extra_postargs=[flagname])
#         except setuptools.distutils.errors.CompileError:
#             return False
#     return True
#
#
# def cpp_flag(compiler):
#     """Return the -std=c++[11/14] compiler flag.
#     The c++14 is prefered over c++11 (when it is available).
#     """
#     if has_flag(compiler, '-std=c++14'):
#         return '-std=c++14'
#     elif has_flag(compiler, '-std=c++11'):
#         return '-std=c++11'
#     else:
#         raise RuntimeError('Unsupported compiler -- at least C++11 support '
#                            'is needed!')
#
#
# class BuildExt(build_ext):
#     """A custom build extension for adding compiler-specific options."""
#     c_opts = {
#         'msvc': ['/EHsc'],
#         'unix': [],
#     }
#
#     if sys.platform == 'darwin':
#         c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
#
#     def build_extensions(self):
#         ct = self.compiler.compiler_type
#         opts = self.c_opts.get(ct, [])
#         if ct == 'unix':
#             opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
#             opts.append(cpp_flag(self.compiler))
#             if has_flag(self.compiler, '-fvisibility=hidden'):
#                 opts.append('-fvisibility=hidden')
#         elif ct == 'msvc':
#             opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
#         for ext in self.extensions:
#             ext.extra_compile_args = opts
#         build_ext.build_extensions(self)
#
#
# setup(
#     name='python_example',
#     version=0.0,
#     author='Sylvain Corlay',
#     author_email='sylvain.corlay@gmail.com',
#     url='https://github.com/pybind/python_example',
#     description='A test project using pybind11',
#     long_description='',
#     ext_modules=ext_modules,
#     cmdclass={'build_ext': BuildExt},
#     zip_safe=False,
# )

# setup(
#     name='video_cap',
#     version='0.0',
#     author='lshi',
#     ext_modules=[
#         Extension("video_capture_cpp",
#                   sources=['video_capture.cpp'],
#                   include_dirs=[".", '/home/lshi/Application/Anaconda/include',
#                                 "/home/lshi/Application/Anaconda/include/opencv/"],
#                   language="c++",
#                   library_dirs=['/home/lshi/Application/Anaconda/lib64/'],
#                   libraries=['opencv_core', 'opencv_cudacodec','opencv_highgui', 'opencv_imgproc'])
#         # extra_compile_args=[
#         #     '-I/home/lshi/Application/Anaconda/include',
#         #     '-I/home/lshi/Application/Anaconda/include/opencv/',
#         #     '-lopencv_cudacodec', '-lopencv_core','-lopencv_highgui', '-lopencv_imgproc',
#         #     '-L/home/lshi/Application/Anaconda/lib64/',
#         # ]),
#     ],
#     cmdclass={'build_ext': build_ext},
# )
