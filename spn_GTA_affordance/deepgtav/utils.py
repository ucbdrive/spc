road_info_dict = {
0:'isonroad',
1:'cannotaccessroadinfo',
2:'forwardVectorX',
3:'forwardVectorY',
4:'forwardVectorZ',
5:'velocityX',
6:'velocityY',
7:'velocityZ',
8:'roadVectorX',
9:'roadVectorY',
10:'roadVectorZ',
11:'leftroadpointX',
12:'leftroadpointY',
13:'leftroadpointZ',
14:'rightroadpointX',
15:'rightroadpointY',
16:'rightroadpointZ',
17:'laneIn',
18:'cantGoLeft',
19:'cantGoRight',
20:'density',
21:'disabled',
22:'highway',
23:'indicateKeepLeft',
24:'indicateKeepRight',
25:'leftTurnsOnly',
26:'noBigVehicles',
27:'noGPS',
28:'offRoad',
29:'slipLane',
30:'special',
31:'maxspeed',
32:'tunnel',
33:'water',
34:'blockIfNoLanes',
35:'gpsBothWays',
36:'narrowRoad',
37:'shortcut',
38:'width'
}

direction_info = {
0:'You Have Arrive',
1:'Recalculating Route, Please make a u-turn where safe',
2: 'Please Proceed the Highlighted Route',
3 : 'Keep Left',
4 : 'In {a} Turn Left',
5 : 'In {a} Turn Right',
6 : 'Keep Right',
7 : 'In {a} Go Straight Ahead',
8 : 'In {a} Join the freeway',
9 : 'In {a} Exit Freeway'
}

def parsedirectioninfo(directioninfo):
    return direction_info[directioninfo[0]].format(a=directioninfo[1]) + ', remaining {b}'.format(b=directioninfo[2])

def parseroadinfo(roadinfo):
    parsedinfo = {}
    for i, info in enumerate(roadinfo):
        parsedinfo[road_info_dict[i]]=info
    parsedinfo['forwardVector'] = (
        parsedinfo.pop('forwardVectorX'), parsedinfo.pop('forwardVectorY'), parsedinfo.pop('forwardVectorZ'))
    parsedinfo['velocity'] = (parsedinfo.pop('velocityX'), parsedinfo.pop('velocityY'), parsedinfo.pop('velocityZ'))
    print(len(parsedinfo))
    if len(parsedinfo)>8:
        parsedinfo['roadVector'] = (
            parsedinfo.pop('roadVectorX'), parsedinfo.pop('roadVectorY'), parsedinfo.pop('roadVectorZ'))
        parsedinfo['leftroadpoint'] = (
        parsedinfo.pop('leftroadpointX'), parsedinfo.pop('leftroadpointY'), parsedinfo.pop('leftroadpointZ'))
        parsedinfo['rightroadpointX'] = (
        parsedinfo.pop('rightroadpointX'), parsedinfo.pop('rightroadpointY'), parsedinfo.pop('rightroadpointZ'))
    return parsedinfo