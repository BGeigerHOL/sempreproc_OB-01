import PyInstaller.__main__

#noarchive=False)
#for d in a.datas:
#    if '_C.cp310-win_amd64.pyd' in d[0]:
#print('***', d)
#a.datas.remove(d)
#break
#





PyInstaller.__main__.run([
    'textcompare.spec',
#    '--onefile',
#    '--noupx',
#    r'--upx-dir=C:\Program Files\upx-3.96-win64',
    '--clean'
])