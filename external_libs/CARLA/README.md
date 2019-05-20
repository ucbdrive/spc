## Install
Download CARLA 0.8.4 [here](https://github.com/carla-simulator/carla/releases/tag/0.8.4)
## Run
### On Ubuntu
`SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh -carla-settings=Example.CarlaSettings.ini -windowed -ResX=256 -ResY=256 -carla-server -carla-no-hud`
### On Windows
`CarlaUE4.exe -windowed -ResX=800 -ResY=600 -carla-server -carla-no-hud -carla-settings=Example.CarlaSettings.ini`
