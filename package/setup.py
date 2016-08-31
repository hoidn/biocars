from setuptools import setup, find_packages

setup(name='biocars',
    version='1.0',
    packages = find_packages('.'),
    package_dir = {'biocars': 'biocars'},
    #package_data = {'biocars': ['data/*']},
    zip_safe=False,
      )
