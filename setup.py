from setuptools import setup, find_packages
import re
from os import path

here = path.abspath(path.dirname(__file__))

# Get all runtime requirements
REQUIRES = []
with open('requirements.txt') as f:
    for line in f:
        line, _, _ = line.partition('#')
        line = line.strip()
        REQUIRES.append(line)

if __name__ == '__main__':
    setup(
        name="PSZS", # Replace with your own username
        version=0.1,
        author="Linus Beckhaus",
        author_email="unrpa@student.kit.edu",
        keywords="domain adaptation, task adaptation, domain generalization, "
                 "transfer learning, deep learning, pytorch",
        description="Fork of THUML Framework for Domain Adaptation in Partially Supervised Zero Shot Setting",
        long_description=open('README.md', encoding='utf8').read(),
        long_description_content_type="text/markdown",
        url="https://github.com/thuml/Transfer-Learning-Library",
        packages=find_packages(exclude=['docs', 'examples','Datasets','Logs', 
                                        'runconfigs', 'checkpoints']),
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries :: Python Modules',
            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        python_requires='>=3.6',
        install_requires=REQUIRES,
        extras_require={
            'dev': [
                'Sphinx',
                'sphinx_rtd_theme',
            ]
        },
    )
