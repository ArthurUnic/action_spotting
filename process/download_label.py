import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(  LocalDirectory="/youtu_pedestrian_detection/zhuhe/soccernet")
mySoccerNetDownloader.password = 's0cc3rn3t'
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])