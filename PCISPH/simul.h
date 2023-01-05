#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "CompactNSearch.h"
using namespace CompactNSearch;

float W(float r);
void initialize(std::vector<glm::vec3>& pos, std::vector<bool>& isW);
bool errCheck(std::vector<float>& dErr, float eta);
float predictDensity(int i, std::vector<glm::vec3>& pos, unsigned int psID, PointSet const& ps);
glm::vec3 CalcPressForce(int i, unsigned int psID, PointSet const& ps, std::vector<glm::vec3>& pos, std::vector<float>& p, std::vector<float>& d, std::vector<bool>& isW);