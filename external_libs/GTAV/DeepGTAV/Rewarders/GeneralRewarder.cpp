#include "GeneralRewarder.h"
#include "SpeedRewarder.h"

GeneralRewarder::GeneralRewarder(LaneRewarder * lanerewarder, float setSpeed, float _agressivity): speedRewarder(setSpeed)
{
	laneRewarder = lanerewarder;
	agressivity = _agressivity;
}

float GeneralRewarder::computeReward(Vehicle vehicle) {
	if (ENTITY::HAS_ENTITY_COLLIDED_WITH_ANYTHING(vehicle))	return -1.0;
	return agressivity*speedRewarder.computeReward(vehicle) + (1.0f - agressivity)*(*laneRewarder).computeReward(vehicle);
}