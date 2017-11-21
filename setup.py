"""
Setup the package.
"""

from setuptools import setup

setup(
    name='spherenet',
    version='0.0.1',
    description='Deep Hyperspherical Learning.',
    long_description='Implementation of https://arxiv.org/abs/1711.03189v1.',
    url='https://github.com/unixpickle/spherenet',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ai neural networks machine learning ml',
    packages=['spherenet'],
    install_requires=['numpy'],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"]
    }
)
