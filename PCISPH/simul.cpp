#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>

// Constants
constexpr float PI = 3.1415926535;
extern float Wall;
extern float h;
extern float m;
extern float deltaTime;
extern float rhoZero;
int Side = (int)(Wall / h) + 4;

void Test(int n) { printf("Test%d\n", n); }
void printPos(glm::vec3 p) { printf("(%.3f, %.3f, %.3f)\n", p.x, p.y, p.z); }
void printFlt(float f) { printf("%.3f\n", f); }

void InitSphere(float SV[1260]) {
	float sphereVerticesIdx[396];
	int count = 0;
	for (float phi = 0.0f; phi <= 180.0f; phi += 30.0f) {
		if (phi == 0.0f || phi == 180.0f) {
			sphereVerticesIdx[count++] = 0.0f;
			sphereVerticesIdx[count++] = h * cos(glm::radians(phi));
			sphereVerticesIdx[count++] = 0.0f;

			continue;
		}

		for (float theta = 0.0f; theta <= 360.0f; theta += 30.0f) {
			sphereVerticesIdx[count++] = h * sin(glm::radians(phi)) * cos(glm::radians(theta));
			sphereVerticesIdx[count++] = h * cos(glm::radians(phi));
			sphereVerticesIdx[count++] = h * sin(glm::radians(phi)) * sin(glm::radians(theta));
		}
	}

	unsigned int Idx[360];
	count = 0;
	int j = 0;
	while (count < 360) {
		for (int i = 1; i <= 12; i++) {
			Idx[count++] = 0;
			Idx[count++] = i + 1;
			Idx[count++] = i;
		}
		for (int i = 1; i <= 51; i++) {
			Idx[count++] = i;
			Idx[count++] = i + 1;
			Idx[count++] = i + 13;

			Idx[count++] = i + 1;
			Idx[count++] = i + 14;
			Idx[count++] = i + 13;

			j++;

			if (j % 12 == 0) {
				i++;
				j = 0;
			}
		}
		for (int i = 53; i <= 64; i++) {
			Idx[count++] = i;
			Idx[count++] = i + 1;
			Idx[count++] = 66;
		}
	}

	count = 0;
	for (int i = 0; i < 360; i++) {
		// position
		SV[count++] = sphereVerticesIdx[3 * Idx[i] + 0];
		SV[count++] = sphereVerticesIdx[3 * Idx[i] + 1];
		SV[count++] = sphereVerticesIdx[3 * Idx[i] + 2];
		// normal
		for (int j = 0; j < 3; j++) SV[count++] = SV[count - 3] / h;
	}
}

float W(float r) 
{
	float hph = 2 * h;
	float m_k = 8 / (PI * hph * hph * hph);
	float res = 0.0f;
	float q = r / hph;

	if (q <= 1.0)
	{
		if (q <= 0.5)
		{
			const float q2 = q * q;
			const float q3 = q2 * q;
			res = m_k * (6.0f * q3 - 6.0f * q2 + 1.0f);
		}
		else
		{
			res = m_k * (2.0 * pow(1.0 - q, 3.0));
		}
	}
	return res;
}

glm::vec3 gradW(glm::vec3 vdiff)
{
	glm::vec3 res;
	float hph = 2 * h;
	float m_l = 48 / (PI * hph * hph * hph);
	const float rl = glm::length(vdiff);
	const float q = rl / hph;
	if ((rl > 1.0e-9) && (q <= 1.0))
	{
		glm::vec3 gradq = vdiff / rl;
		gradq /= hph;

		if (q <= 0.5)
		{
			res = m_l * q * (3.0f * q - 2.0f) * gradq;
		}
		else
		{
			const float factor = 1.0f - q;
			res = m_l * (-factor * factor) * gradq;
		}
	}
	else
		res = glm::vec3(0);

	return res;
};

void initialize(std::vector<glm::vec3>& pos, std::vector<bool>& isW)
{
	int sign = -1;
	float interval = 4 * h / 3;
	for (float dx = -Wall-4*h; dx < Wall+4*h; dx += interval)
		for (float dz = -Wall - 4 * h; dz < Wall + 4 * h; dz += interval)
			for (float dy = -4 * h; dy < 2.f; dy += interval)
			{

				bool isBoundary = abs(dx) > Wall || abs(dz) > Wall || dy < 0.0f;

				if (isBoundary)
				{
					pos.push_back(glm::vec3(dx, dy, dz));
					isW.push_back(false);
				}
				else if (dy <= 0.1f && abs(dx) <= Wall && abs(dz) <= Wall)
				{
					pos.push_back(glm::vec3(dx, dy, dz));
					isW.push_back(true);
				}
				else if (dy >= 0.8f && dy <= 1.f && abs(dx) <= 0.2f && abs(dz) <= 0.2f)
				{
					pos.push_back(glm::vec3(dx, dy, dz));
					isW.push_back(true);
				}
			}
}

bool errCheck(std::vector<float>& dErr, float eta) 
{
	for (int i = 0; i < dErr.size(); i++) {
		if (dErr[i] > eta)
			return true;
	}
	return false;
}

glm::vec3 findCellNum(glm::vec3 pos) 
{
	int xIdx = (int)floor((pos[0] + Wall + 4 * h) / (2 * h));
	int yIdx = (int)floor((pos[1] + 4 * h) / (2 * h));
	int zIdx = (int)floor((pos[2] + Wall + 4 * h) / (2 * h));
	return glm::vec3(xIdx, yIdx, zIdx);
}

std::vector<std::vector<int>> create_Particle_List_in_Cell(std::vector<glm::vec3>& pos) 
{
	std::vector<std::vector<int>> cList(Side * Side * Side);

	for (int i = 0; i < pos.size(); i++)
	{
		glm::vec3 CNV = findCellNum(pos[i]);
		int cIdx = Side * Side * CNV.y + Side * CNV.x + CNV.z;
		if (cIdx < 0 || cIdx >= Side * Side * Side) continue;
		cList[cIdx].push_back(i);
	}

	return cList;
}

std::vector<std::vector<int>> createNeighborList(std::vector<glm::vec3>& pos, std::vector<bool>& isW)
{
	std::vector<std::vector<int>> nList;
	std::vector<std::vector<int>> cList = create_Particle_List_in_Cell(pos);

	for (int i = 0; i < pos.size(); i++)
	{
		glm::vec3 CNV = findCellNum(pos[i]);
		std::vector<int> Neighb;

		for (int a = -1; a <= 1; a++)
			for (int b = -1; b <= 1; b++)
				for (int c = -1; c <= 1; c++)
				{
					int cIdx = Side * Side * (CNV.y + a) + Side * (CNV.x + b) + (CNV.z + c);
					if (cIdx < 0 || cIdx >= Side * Side * Side) continue;

					Neighb.insert(Neighb.end(), cList[cIdx].begin(), cList[cIdx].end());
				}

		nList.push_back(Neighb);
	}

	return nList;
}

float predictDensity(int i, std::vector<glm::vec3>& pos, std::vector<std::vector<int>>& nList)
{
	float rho = 0.0f;
	glm::vec3 iPos = pos[i];

	for (int n = 0; n < nList[i].size(); n++)
	{
		if (i == nList[i][n]) continue;

		glm::vec3 nPos = pos[nList[i][n]];
		float dist = glm::length(iPos - nPos);
		rho += m * W(dist);
	}

	return rho;
}

int getParticleCount(std::vector<bool>& isW)
{
	int count = 0;
	for (int i = 0; i < isW.size(); i++)
		if (isW[i]) count++;
	return count;
}

glm::vec3 CalcPressForce(int i, int n, std::vector<glm::vec3>& pos, std::vector<float>& p, std::vector<float>& d, std::vector<bool>& isW)
{
	glm::vec3 nablaW = gradW(pos[i] - pos[n]);

	float coeff;
	//coeff = m * m * (p[i] / (d[i] * d[i]) + p[n] / (d[n] * d[n]));
	if (isW[n])
		coeff = m * m * (p[i] + (d[n] / rhoZero) * p[n]) / rhoZero;
	else
		coeff = m * m * p[i] / rhoZero;
	return (coeff * nablaW);
}
