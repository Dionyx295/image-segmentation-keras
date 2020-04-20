# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['skyeye_segmentation\\entrypoint.py'],
             pathex=['C:\\Users\\Valentin MAURICE\\PycharmProjects\\image-segmentation-keras'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.datas += Tree('venv/Lib/site-packages/scipy', prefix='scipy')
a.datas += Tree('venv/Lib/site-packages/sklearn', prefix='sklearn')
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='entrypoint',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True , icon='Eye.ico')
