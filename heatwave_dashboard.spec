# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['heatwave_dashboard.py'],
    pathex=[],
    binaries=[],
    datas=[('Heatwave_Training_Data_With_RoadDensity (4).csv', '.'), ('DistrictBoundary_SHP', 'DistrictBoundary_SHP'), ('heatwave_vulnerability_points.gpkg', '.'), ('district_boundaries.shp', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='heatwave_dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='heatwave_dashboard.app',
    icon=None,
    bundle_identifier=None,
)
