#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>

#include "CompactNSearch.h"
using namespace CompactNSearch;

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
	for (float dx = -Wall - 4 * h; dx < Wall + 4 * h; dx += interval)
		for (float dz = -Wall - 4 * h; dz < Wall + 4 * h; dz += interval)
			for (float dy = -4 * h; dy < 2.f ; dy += interval)
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
				else if (dy >= 0.5f && dy <= 1.0f && abs(dx) <= 0.3f && abs(dz) <= 0.3f)
				{
					pos.push_back(glm::vec3(dx, dy, dz));
					isW.push_back(true);
				}
				//else if (dy <= 0.5f && dx <= 0.f)
				//{
				//	pos.push_back(glm::vec3(dx, dy, dz));
				//	isW.push_back(true);
				//}
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

float predictDensity(int i, std::vector<glm::vec3>& pos, unsigned int psID, PointSet const& ps)
{
	float rho = 0.0f;
	glm::vec3 iPos = pos[i];

	#pragma omp parallel
	for (unsigned int j = 0; j < ps.n_neighbors(psID, i); j++)
	{
		int n = ps.neighbor(psID, i, j);
		if (i == n) continue;

		glm::vec3 nPos = pos[n];
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

glm::vec3 CalcPressForce(int i, unsigned int psID, PointSet const& ps, std::vector<glm::vec3>& pos, std::vector<float>& p, std::vector<float>& d, std::vector<bool>& isW)
{
	glm::vec3 totalFpi(0);

	#pragma omp parallel
	for (unsigned int j = 0; j < ps.n_neighbors(psID, i); j++)
	{
		int n = ps.neighbor(psID, i, j);
		if (i == n) continue;

		glm::vec3 nablaW = gradW(pos[i] - pos[n]);

		float coeff;
		//coeff = m * m * (p[i] / (d[i] * d[i]) + p[n] / (d[n] * d[n]));
		if (isW[n])
			coeff = m * m * (p[i] + (d[n] / rhoZero) * p[n]) / rhoZero;
		else
			coeff = m * m * p[i] / rhoZero;

		totalFpi += coeff * nablaW;
	}

	return totalFpi;
}