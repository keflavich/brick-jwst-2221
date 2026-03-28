Photometry Products and Filename Reference
=========================================

This page is a reference for pipeline outputs and filename structure.

For run order, commands, and operations, see
``photometry_operator_runbook.rst``.


Common filename tokens
----------------------

Many outputs share tokenized suffixes:

- ``{desat}`` is ``_unsatstar`` when desaturated inputs are used, else empty.
- ``{bgsub}`` is ``_bgsub`` when background subtraction is enabled, else empty.
- ``{epsf_}`` is ``_epsf`` when ePSF building is enabled, else empty.
- ``{blur_}`` is ``_blur`` when PSF blur mode is enabled, else empty.
- ``{group}`` is ``_group`` when grouped DAOPHOT fitting is enabled, else empty.
- ``{visitid_}`` is ``_visitNNN`` for per-exposure runs.
- ``{vgroupid_}`` is ``_vgroupNNNN`` for per-exposure runs.
- ``{exposure_}`` is ``_expNNNNN`` for per-exposure runs.


crowdsource_catalogs_long
-------------------------

Module:
``brick2221/analysis/crowdsource_catalogs_long.py``

Input image patterns
~~~~~~~~~~~~~~~~~~~~

- Mosaic mode:

  - ``{basepath}/{FILTER}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filter}-{module}_i2d.fits``
  - fallbacks include ``*_realigned-to-refcat.fits`` and ``*_i2d_unsatstar.fits``

- Per-exposure mode (``--each-exposure``):

  - ``{basepath}/{FILTER}/pipeline/jw0{proposal_id}{field}{visitid}*{module}*{each_suffix}.fits``

Crowdsource outputs
~~~~~~~~~~~~~~~~~~~

- Star table:
  ``{basepath}/{FILTER}/{filter}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_crowdsource_{suffix}.fits``

- Sky/model table (two HDUs):
  ``{basepath}/{FILTER}/{filter}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_crowdsource_skymodel_{suffix}.fits``

- Optional PSF stamp:
  ``{basepath}/{FILTER}/{filter}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_crowdsource_{suffix}_psf.fits``

Typical suffixes include ``unweighted`` and ``nsky0``.

DAOPHOT outputs (when ``--daophot``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Catalogs:

  - ``..._daophot_basic.fits``
  - ``..._daophot_iterative.fits``

- Model/residual images:

  - ``..._daophot_basic_model.fits``
  - ``..._daophot_basic_residual.fits``
  - ``..._daophot_iterative_model.fits``
  - ``..._daophot_iterative_residual.fits``

- Optional ePSF products:

  - ``..._daophot_epsf.fits``
  - ``..._daophot_epsf.png``

- Background intermediates when ``--bgsub``:

  - ``*_background.fits``
  - ``*_bgsub.fits``


merge_catalogs
--------------

Module:
``brick2221/analysis/merge_catalogs.py``

Per-exposure single-filter merge outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``merge_individual_frames`` into ``{basepath}/catalogs/``:

- ``{filter}_{module}_indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_{method}{suffix}_allcols.fits``
- ``{filter}_{module}_indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_{method}{suffix}.fits``

Cross-filter merged outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``merge_catalogs`` into ``{basepath}/catalogs/``:

- ``{catalog_type}_{module}{_indivexp_if_set}_photometry_tables_merged{desat}{bgsub}{epsf_}{blur_}.fits``
- ``{catalog_type}_{module}{_indivexp_if_set}_photometry_tables_merged{desat}{bgsub}{epsf_}{blur_}.ecsv``
- ``..._qualcuts.fits``
- ``..._qualcuts_oksep2221.fits``

Saturated-star dependency
~~~~~~~~~~~~~~~~~~~~~~~~~

Replacement/flagging functions expect:

- ``{basepath}/{FILTER}/pipeline/jw0{proposal}-o{obsnum}_t001_nircam_clear-{filter}-merged_i2d_satstar_catalog.fits``


saturated_star_finding
----------------------

Module:
``brick2221/reduction/saturated_star_finding.py``

Input search patterns
~~~~~~~~~~~~~~~~~~~~~

- NIRCam: ``/orange/adamginsburg/jwst/{target}/{FILTER}/pipeline/*{module}*align*crf.fits``
- MIRI: ``/orange/adamginsburg/jwst/{target}/{FILTER}/pipeline/*mirimage_cal.fits``

Output
~~~~~~

- ``{input_filename_without_.fits}_satstar_catalog.fits``

Compatibility note
~~~~~~~~~~~~~~~~~~

``merge_catalogs.py`` expects merged-style saturated-star filenames
(``*_merged_i2d_satstar_catalog.fits``), while
``saturated_star_finding.py`` writes per-input filenames
(``*_satstar_catalog.fits``). A rename/symlink/wrapper mapping step may be
needed in downstream processing.
