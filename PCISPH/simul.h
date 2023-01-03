#pragma once

#include <vector>
#include <glm/glm.hpp>


void Test(int n);
void printPos(glm::vec3 p);
void printFlt(float f);

void InitSphere(float SV[1260]);
float W(float r);
void initialize(std::vector<glm::vec3>& pos, std::vector<bool>& isW);
bool errCheck(std::vector<float>& dErr, float eta);
std::vector<std::vector<int>> create_Particle_List_in_Cell(std::vector<glm::vec3>& pos);
std::vector<std::vector<int>> createNeighborList(std::vector<glm::vec3>& pos, std::vector<bool>& isW);
float predictDensity(int i, std::vector<glm::vec3>& pos, std::vector<std::vector<int>>& nList);
int getParticleCount(std::vector<bool>& isW);
glm::vec3 CalcPressForce(int i, int n, std::vector<glm::vec3>& pos, std::vector<float>& p, std::vector<float>& d, std::vector<bool>& isW);
