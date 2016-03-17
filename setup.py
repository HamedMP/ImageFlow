__author__ = 'HANEL'

from setuptools import setup

setup(name='imageflow',
      version='0.0.2',

      description='Import, Convert (and Soon Train) images with TensorFlow',

      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
      ],
      keywords='tensorflow image cnn',

      url='http://hamedmp.github.io/ImageFlow/',

      author='Hamed Mohammadpour',
      author_email='hamedmp@my.com',

      license='MIT',

      packages=['imageflow'],
      zip_safe=False,

      install_requires=['numpy', 'Pillow'],

      include_package_data=True,

      dependency_links=[''],

      )
