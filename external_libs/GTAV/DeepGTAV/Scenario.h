#pragma once

#include <stdlib.h>
#include <ctime>

#include "lib/script.h"
#include "lib/utils.h"

#include "lib/rapidjson/document.h"
#include "lib/rapidjson/stringbuffer.h"

#include "ScreenCapturer.h"
#include "Rewarders\Rewarder.h"
#include "Rewarders\GeneralRewarder.h"
#include "Rewarders\LaneRewarder.h"

using namespace rapidjson;

//#define DEBUG 1

class Scenario {
private:
	bool is_slow_motion = true;
	float slow_rate = 6;

	static char* weatherList[14];
	static char* vehicleList[3];

	Vehicle vehicle = NULL;
	Player player = NULL;
	Ped ped = NULL;
	Cam camera = NULL;
	Vector3 dir;

	float x, y;
	int hour, minute;
	const char* _weather;
	const char* _vehicle;
	int width, height;

	bool vehicles;
	bool peds;
	bool trafficSigns; //TODO
	bool direction;
	bool reward;
	bool throttle;
	bool brake;
	bool steering;
	bool speed;
	bool yawRate;
	bool drivingMode; //TODO
	bool location;
	bool time;
	bool roadinfo;

	float currentThrottle = 0.0;
	float currentBrake = 0.0;
	float currentSteering = 0.0;

	GeneralRewarder* rewarder;
	std::clock_t lastSafetyCheck;
	int _drivingMode;
	float _setSpeed;

	bool running = false;
	Document d;

public:
	int rate;
	void start(const Value & sc, const Value & dc, LaneRewarder * lanerewarder);
	void config(const Value & sc, const Value & dc, LaneRewarder * lanerewarder);
	void stop();
	void setCommands(float throttle, float brake, float steering);
	void run();

	ScreenCapturer* screenCapturer;
	StringBuffer generateMessage();

private:
	void parseScenarioConfig(const Value& sc, bool setDefaults);
	void parseDatasetConfig(const Value& dc, bool setDefaults, LaneRewarder * lanerewarder);
	void buildScenario();

	void setVehiclesList();
	void setPedsList();
	void setTrafficSignsList();
	void setDirection();
	void setReward();
	void setThrottle();
	void setBrake();
	void setSteering();
	void setSpeed();
	void setYawRate();
	void setDrivingMode();
	void setLocation();
	void setTime();
	void setroadinfo();
};