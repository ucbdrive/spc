#pragma once

#include "Rewarder.h"
#include "LaneRewarder.h"
#include "SpeedRewarder.h"

class GeneralRewarder : public Rewarder {
public:
	SpeedRewarder speedRewarder;
	LaneRewarder* laneRewarder;
	float agressivity;
	GeneralRewarder(LaneRewarder* lanerewarder, float setSpeed, float _agressivity);
	float computeReward(Vehicle vehicle);
};