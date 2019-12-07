mypath = 'F:\weather_deter\Image\cl500'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for n in range(1, len(onlyfiles)):
    im = cv2.imread(join(mypath, onlyfiles[n]), 0)
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        """
