Supposedly have to use a different version of python to run [qsofitmore](https://github.com/rudolffu/qsofitmore/blob/main/README.md)


```
(denison) o_thorp@Olivers-MacBook-Air denison_2026 % conda install -c conda-forge dustmaps
2 channel Terms of Service accepted
Retrieving notices: done
Channels:
 - conda-forge
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: failed

LibMambaUnsatisfiableError: Encountered problems while solving:
  - nothing provides _python_rc needed by python-3.14.0rc2-h6ea10a9_0_cp314t

Could not solve for environment specs
The following packages are incompatible
├─ dustmaps =* * is installable with the potential options
│  ├─ dustmaps [1.0.10|1.0.11|1.0.12|1.0.13|1.0.14] would require
│  │  └─ healpy =* * with the potential options
│  │     ├─ healpy [1.15.0|1.15.1|...|1.19.0] would require
│  │     │  └─ python >=3.10,<3.11.0a0 *_cpython, which can be installed;
│  │     ├─ healpy 1.16.1 would require
│  │     │  └─ numpy >=1.23.4,<2.0a0 * with the potential options
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1] would require
│  │     │     │  └─ python >=3.10,<3.11.0a0 *_cpython, which can be installed;
│  │     │     ├─ numpy [1.23.5|1.24.0|...|1.26.4], which can be installed;
│  │     │     ├─ numpy [1.23.4|1.23.5|...|1.24.4] would require
│  │     │     │  └─ python >=3.8,<3.9.0a0 *_cpython, which can be installed;
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1] would require
│  │     │     │  └─ python >=3.9,<3.10.0a0 *_cpython, which can be installed;
│  │     │     ├─ numpy [1.26.0|1.26.2|1.26.3|1.26.4|2.0.1] would require
│  │     │     │  └─ python [>=3.12,<3.13.0a0 *_cpython|>=3.12.0rc3,<3.13.0a0 *_cpython], which can be installed;
│  │     │     ├─ numpy 1.23.4, which can be installed;
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1] would require
│  │     │     │  └─ python >=3.10,<3.11.0a0 *, which can be installed;
│  │     │     ├─ numpy [1.23.4|1.23.5|1.24.3] would require
│  │     │     │  └─ python >=3.8,<3.9.0a0 *, which can be installed;
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1] would require
│  │     │     │  └─ python >=3.9,<3.10.0a0 *, which can be installed;
│  │     │     └─ numpy [1.26.0|1.26.2|1.26.3|1.26.4|2.0.1] would require
│  │     │        └─ python >=3.12,<3.13.0a0 *, which can be installed;
│  │     ├─ healpy [1.15.0|1.15.1|...|1.16.5] would require
│  │     │  └─ python >=3.8,<3.9.0a0 *_cpython, which can be installed;
│  │     ├─ healpy [1.15.0|1.15.1|...|1.17.3] would require
│  │     │  └─ python >=3.9,<3.10.0a0 *_cpython, which can be installed;
│  │     ├─ healpy 1.16.2 would require
│  │     │  └─ numpy >=1.23.5,<2.0a0 * with the potential options
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  │     │     ├─ numpy [1.23.5|1.24.0|...|1.26.4], which can be installed;
│  │     │     ├─ numpy [1.23.4|1.23.5|...|1.24.4], which can be installed (as previously explained);
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  │     │     ├─ numpy [1.26.0|1.26.2|1.26.3|1.26.4|2.0.1], which can be installed (as previously explained);
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  │     │     ├─ numpy [1.23.4|1.23.5|1.24.3], which can be installed (as previously explained);
│  │     │     ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  │     │     └─ numpy [1.26.0|1.26.2|1.26.3|1.26.4|2.0.1], which can be installed (as previously explained);
│  │     ├─ healpy 1.16.5 would require
│  │     │  └─ libgfortran5 >=12.3.0 * with the potential options
│  │     │     ├─ libgfortran5 12.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 *_0, which can be installed;
│  │     │     ├─ libgfortran5 12.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 12_3_0_*_1, which can be installed;
│  │     │     ├─ libgfortran5 12.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 12_3_0_*_2, which can be installed;
│  │     │     ├─ libgfortran5 12.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 12_3_0_*_3, which can be installed;
│  │     │     ├─ libgfortran5 12.4.0 would require
│  │     │     │  └─ libgfortran ==12.4.0 *, which can be installed;
│  │     │     ├─ libgfortran5 12.4.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 12_4_0_*_0, which can be installed;
│  │     │     ├─ libgfortran5 12.4.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 12_4_0_*_1, which can be installed;
│  │     │     ├─ libgfortran5 12.4.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 12_4_0_*_2, which can be installed;
│  │     │     ├─ libgfortran5 12.4.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 12_4_0_*_103, which can be installed;
│  │     │     ├─ libgfortran5 13.2.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 13_2_0_*_1, which can be installed;
│  │     │     ├─ libgfortran5 13.2.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 13_2_0_*_2, which can be installed;
│  │     │     ├─ libgfortran5 13.2.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 13_2_0_*_3, which can be installed;
│  │     │     ├─ libgfortran5 13.3.0 would require
│  │     │     │  └─ libgfortran ==13.3.0 *, which can be installed;
│  │     │     ├─ libgfortran5 13.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 13_3_0_*_0, which can be installed;
│  │     │     ├─ libgfortran5 13.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 13_3_0_*_1, which can be installed;
│  │     │     ├─ libgfortran5 13.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 13_3_0_*_2, which can be installed;
│  │     │     ├─ libgfortran5 13.3.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 13_3_0_*_103, which can be installed;
│  │     │     ├─ libgfortran5 13.4.0 would require
│  │     │     │  └─ libgfortran ==13.4.0 *, which can be installed;
│  │     │     ├─ libgfortran5 14.2.0 would require
│  │     │     │  └─ libgfortran ==14.2.0 *, which can be installed;
│  │     │     ├─ libgfortran5 14.2.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 14_2_0_*_0, which can be installed;
│  │     │     ├─ libgfortran5 14.2.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 14_2_0_*_1, which can be installed;
│  │     │     ├─ libgfortran5 14.2.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 14_2_0_*_2, which can be installed;
│  │     │     ├─ libgfortran5 14.2.0 would require
│  │     │     │  └─ libgfortran ==5.0.0 14_2_0_*_103, which can be installed;
│  │     │     ├─ libgfortran5 14.3.0 would require
│  │     │     │  └─ libgfortran ==14.3.0 *, which can be installed;
│  │     │     ├─ libgfortran5 15.1.0 would require
│  │     │     │  └─ libgfortran ==15.1.0 *, which can be installed;
│  │     │     └─ libgfortran5 15.2.0 would require
│  │     │        └─ libgfortran ==15.2.0 *, which can be installed;
│  │     ├─ healpy [1.16.6|1.17.1|1.17.3|1.18.0] would require
│  │     │  └─ libgfortran5 >=13.2.0 *, which can be installed (as previously explained);
│  │     ├─ healpy [1.16.6|1.17.1|...|1.19.0] would require
│  │     │  └─ python >=3.12,<3.13.0a0 *_cpython, which can be installed;
│  │     ├─ healpy [1.17.3|1.18.0|1.18.1|1.19.0] would require
│  │     │  └─ python [>=3.13,<3.14.0a0 *_cp313|>=3.13.0rc2,<3.14.0a0 *_cp313], which can be installed;
│  │     ├─ healpy 1.18.1 would require
│  │     │  └─ libgfortran5 >=15.1.0 *, which can be installed (as previously explained);
│  │     ├─ healpy 1.18.1 would require
│  │     │  └─ libgfortran5 >=14.2.0 *, which can be installed (as previously explained);
│  │     ├─ healpy 1.18.1 would require
│  │     │  └─ python [>=3.14.0rc2,<3.15.0a0 *|>=3.14.0rc3,<3.15.0a0 *] with the potential options
│  │     │     ├─ python [3.14.0|3.14.1|3.14.2], which can be installed;
│  │     │     └─ python [3.14.0rc2|3.14.0rc3] would require
│  │     │        └─ _python_rc =* *, which does not exist (perhaps a missing channel);
│  │     ├─ healpy 1.19.0 would require
│  │     │  └─ libgfortran5 >=14.3.0 *, which can be installed (as previously explained);
│  │     ├─ healpy 1.19.0 would require
│  │     │  └─ python >=3.14,<3.15.0a0 *, which can be installed;
│  │     ├─ healpy [1.14.0|1.15.0] would require
│  │     │  └─ python >=3.8,<3.9.0a0 *, which can be installed;
│  │     └─ healpy [1.14.0|1.15.0] would require
│  │        └─ python >=3.9,<3.10.0a0 *, which can be installed;
│  ├─ dustmaps [1.0.10|1.0.11|...|1.0.9] would require
│  │  └─ python >=3.10,<3.11.0a0 *_cpython, which can be installed;
│  ├─ dustmaps [1.0.10|1.0.11|1.0.12|1.0.13|1.0.9] would require
│  │  └─ python >=3.8,<3.9.0a0 *_cpython, which can be installed;
│  ├─ dustmaps [1.0.10|1.0.11|...|1.0.9] would require
│  │  └─ python >=3.9,<3.10.0a0 *_cpython, which can be installed;
│  ├─ dustmaps [1.0.12|1.0.13|1.0.14] would require
│  │  └─ python >=3.12,<3.13.0a0 *_cpython, which can be installed;
│  ├─ dustmaps [1.0.13|1.0.14] would require
│  │  └─ python >=3.13,<3.14.0a0 *_cp313, which can be installed;
│  ├─ dustmaps 1.0.14 would require
│  │  └─ python >=3.14.0rc2,<3.15.0a0 * with the potential options
│  │     ├─ python [3.14.0|3.14.1|3.14.2], which can be installed;
│  │     └─ python [3.14.0rc2|3.14.0rc3], which cannot be installed (as previously explained);
│  ├─ dustmaps [1.0.7|1.0.8|1.0.9] would require
│  │  └─ python >=3.8,<3.9.0a0 *, which can be installed;
│  └─ dustmaps [1.0.7|1.0.8|1.0.9] would require
│     └─ python >=3.9,<3.10.0a0 *, which can be installed;
├─ libgfortran5 ==11.3.0 * is installable with the potential options
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_27, which can be installed;
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_28, which can be installed;
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_29, which can be installed;
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_30, which can be installed;
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_31, which can be installed;
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_32, which can be installed;
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_24, which can be installed;
│  ├─ libgfortran5 11.3.0 would require
│  │  └─ libgfortran ==5.0.0 *_25, which can be installed;
│  └─ libgfortran5 11.3.0 would require
│     └─ libgfortran ==5.0.0 *_26, which can be installed;
├─ libgfortran ==5.0.0 * is not installable because it conflicts with any installable versions previously reported;
├─ numpy-base ==2.0.1 * is installable with the potential options
│  ├─ numpy-base 2.0.1 would require
│  │  └─ python >=3.10,<3.11.0a0 *, which can be installed;
│  ├─ numpy-base 2.0.1 would require
│  │  └─ numpy ==2.0.1 py311he598dae_1, which can be installed;
│  ├─ numpy-base 2.0.1 would require
│  │  └─ python >=3.12,<3.13.0a0 *, which can be installed;
│  └─ numpy-base 2.0.1 would require
│     └─ python >=3.9,<3.10.0a0 *, which can be installed;
├─ numpy ==2.0.1 * is installable with the potential options
│  ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  ├─ numpy [1.26.0|1.26.2|1.26.3|1.26.4|2.0.1], which can be installed (as previously explained);
│  ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  ├─ numpy [1.23.4|1.23.5|...|2.0.1], which can be installed (as previously explained);
│  ├─ numpy [1.26.0|1.26.2|1.26.3|1.26.4|2.0.1], which can be installed (as previously explained);
│  └─ numpy 2.0.1 conflicts with any installable versions previously reported;
└─ pin on python 3.11.* =* * is not installable because it requires
   └─ python =3.11 *, which conflicts with any installable versions previously reported.

Pins seem to be involved in the conflict. Currently pinned specs:
 - python=3.11
```