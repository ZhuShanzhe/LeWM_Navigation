"""Fetch native Jev vision sources/assets, without starting GPU services."""
from instance_fetch_assets import ROOT,source,download
for repo,name in [('ChaoningZhang/MobileSAM','MobileSAM'),('WongKinYiu/yolov7','yolov7')]:
    source(repo,name)
download('https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt','yolov7-e6e.pt')
