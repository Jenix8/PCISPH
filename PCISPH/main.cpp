#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include "simul.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include "CompactNSearch.h"
using namespace CompactNSearch;
using namespace std;
using namespace chrono;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// initializing setup
const unsigned int WIDTH = 1280;
const unsigned int HEIGHT = 720;
constexpr float PI = 3.1415926535;

// camera setup
Camera camera(glm::vec3(3.0f, 2.0f, 3.0f));
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

// visualizing
//glm::vec3 waterCol(0.11f, 0.64f, 0.96f);
glm::vec3 waterCol(0.11f, 0.35f, 0.6f);
glm::vec3 whiteCol(1.f, 1.f, 1.f);
glm::vec3 highPres(0.7f, 0.f, 0.f);

// particle setting
float WallX = 0.5f;
float WallZ = 0.5f;
float h = 0.025f;					// particle radius
float rhoZero = 1000;				// reference density
float m = rhoZero * 4 * PI * h * h * h / 3;	// particle mass (all particles have equal masses)
glm::vec3 g(0.0f, -9.81f, 0.0f);	// gravity
glm::vec3 Fg = m * g;

std::vector<glm::vec3> x;			// position
std::vector<std::array<Real, 3>> Nx;	// positions for neighbor search
std::vector<int> water;				// waterIdx
std::vector<glm::vec3> v;			// velocity
std::vector<bool> isWat;			// true: water, false: boundary
std::vector<glm::vec3> px;			// predicted position
std::vector<glm::vec3> pv;			// predicted velocity
std::vector<float> d;				// density
std::vector<float> dErr;			// density variation
float delta = 0.003f;
float eta = 0.05f;

// another 
enum class VISUAL_MODE
{
	VELOCITY,
	PRESSURE,
	DENSITY
};

float pVertex[4000000];
int minIterations = 3;
float deltaTime = 1 / 400.f;
int frame = 0;
VISUAL_MODE mode = VISUAL_MODE::VELOCITY;

int main() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "PCISPH simulator", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << '\n';
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetScrollCallback(window, scroll_callback);
	//glfwSetCursorPosCallback(window, mouse_callback);					  
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);		  

	glewExperimental = GL_TRUE;

	GLenum errorCode = glewInit();
	if (GLEW_OK != errorCode) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_PROGRAM_POINT_SIZE);
	//glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Shader pointShader("Point.vs", "Point.fs");

	initialize(x, isWat);
	int allPtc = x.size();

	Nx.resize(allPtc);
	v.resize(allPtc);
	px = x;
	pv = v;
	d.resize(allPtc);
	dErr.resize(allPtc);

	for (int i = 0; i < allPtc; i++)
		if (isWat[i]) water.push_back(i);
		else d[i] = rhoZero;

	int pCount = water.size();
	printf("pCount: %d\n", pCount);

	for (int i = 0; i < allPtc; i++)
		for (int j = 0; j < 3; j++)
			Nx[i][j] = x[i][j];

	NeighborhoodSearch nsearch(2 * h);

	unsigned int psID = nsearch.add_point_set(Nx.front().data(), Nx.size());
	nsearch.z_sort();
	nsearch.find_neighbors();
	PointSet const& ps = nsearch.point_set(psID);

	unsigned int pVBO, pVAO;
	glGenVertexArrays(1, &pVAO);
	glGenBuffers(1, &pVBO);
	glBindBuffer(GL_ARRAY_BUFFER, pVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pVertex), pVertex, GL_STATIC_DRAW);
	glBindVertexArray(pVAO);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	pointShader.use();
	glBindVertexArray(pVAO);
	glBindBuffer(GL_ARRAY_BUFFER, pVBO);

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		// 1. DEFAULT SETTING
		frame++;
		processInput(window);
		glClearColor(0.6f, 0.5f, 0.4f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// 2. ANIMATING
		system_clock::time_point T1 = system_clock::now();

		#pragma omp parallel default(shared)
		{
			// update only water particle
			#pragma omp for schedule(static)
			for (int i = 0; i < pCount; i++)
				for (int j = 0; j < 3; j++)
					Nx[water[i]][j] = x[water[i]][j];
		}

		nsearch.find_neighbors();
		nsearch.update_point_sets();

		system_clock::time_point T2 = system_clock::now();

		// initialize pressure and pressure force as 0
		std::vector<float> p(allPtc);			// pressure
		std::vector<glm::vec3> Fp(allPtc);		// pressure Force
		std::vector<glm::vec3> accel(allPtc);	// acceleration

		// START PREDICTING LOOP
		int Iter = 0;
		while (errCheck(dErr, eta) || Iter < minIterations)
		{
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (int pIdx = 0; pIdx < pCount; pIdx++)
				{
					int i = water[pIdx];

					// predict next v and x
					accel[i] = (Fg + Fp[i]) / m;
					pv[i] = v[i] + deltaTime * accel[i];
					px[i] = x[i] + deltaTime * pv[i];
				}

				#pragma omp for schedule(static)
				for (int pIdx = 0; pIdx < pCount; pIdx++)
				{
					int i = water[pIdx];

					// predict next density and its variation
					d[i] = predictDensity(i, px, psID, ps);
					dErr[i] = d[i] - rhoZero;

					// update pressure
					p[i] += dErr[i] * delta;
					p[i] = std::max(p[i], 0.f);
				}

				#pragma omp for schedule(static)
				for (int pIdx = 0; pIdx < pCount; pIdx++)
				{
					int i = water[pIdx];
					// compute Fp
					#pragma omp parallel
					Fp[i] -= CalcPressForce(i, psID, ps, px, p, d, isWat);
				}
			}

			Iter++;
		}
		
		v = pv;
		x = px;

		system_clock::time_point T3 = system_clock::now();

		// 3. RENDERING
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera.GetViewMatrix();
		pointShader.setMat4("projection", projection);
		pointShader.setMat4("view", view);

		#pragma omp parallel default(shared)
		{
			for (int i = 0; i < allPtc; i++)
			{
				pVertex[7 * i + 0] = x[i].x;
				pVertex[7 * i + 1] = x[i].y;
				pVertex[7 * i + 2] = x[i].z;
				pVertex[7 * i + 6] = 1;

				glm::vec3 color;
				if (!isWat[i])
				{
					if (x[i][0] >= WallX) pVertex[7 * i + 6] = 0;
					if (x[i][2] >= WallZ) pVertex[7 * i + 6] = 0;
					//pVertex[7 * i + 6] = 0;
					color = glm::vec3(0.2f);
				}
				else
				{
					pVertex[7 * i + 6] = 0.7f;

					if (mode == VISUAL_MODE::VELOCITY)
					{
						float vel = glm::length(v[i]);
						color = vel * (whiteCol - waterCol) / 4.f + waterCol;
					}
					else if (mode == VISUAL_MODE::PRESSURE)
					{
						color = p[i] * (highPres - waterCol) / 2.f + waterCol;
					}
					else if (mode == VISUAL_MODE::DENSITY)
					{
						if (d[i] < rhoZero)
							color = waterCol;
						else
							color = (d[i] - rhoZero) * (highPres - waterCol) / (rhoZero * eta) + waterCol;
					}
				}

				pVertex[7 * i + 3] = color.x;
				pVertex[7 * i + 4] = color.y;
				pVertex[7 * i + 5] = color.z;
			}
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(pVertex), pVertex, GL_STATIC_DRAW);
		glDrawArrays(GL_POINTS, 0, allPtc);

		system_clock::time_point T4 = system_clock::now();

		if (false)
		{
			printf("\n%d's Iterations: %d\n", frame, Iter);
			cout << "NeighborSearch\t:" << duration_cast<microseconds>(T2 - T1).count() << "\tmicros" << endl;
			cout << "Predicting\t:" << duration_cast<microseconds>(T3 - T2).count() << "\tmicros" << endl;
			cout << "Rendering\t:" << duration_cast<microseconds>(T4 - T3).count() << "\tmicros" << endl;
			cout << "frame time\t:" << duration_cast<microseconds>(T4 - T1).count() / 1000.f << "\tms\n" << endl;
		}
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glDeleteVertexArrays(1, &pVAO);
	glDeleteBuffers(1, &pVBO);

	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}
void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	const float cameraSpeed = static_cast<float>(2.5f * deltaTime);
	float T = 10 * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, T);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, T);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, T);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, T);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		camera.ProcessKeyboard(UPWARD, T);
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		camera.ProcessKeyboard(DOWNWARD, T);

	if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
		mode = VISUAL_MODE::VELOCITY;
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
		mode = VISUAL_MODE::PRESSURE;
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
		mode = VISUAL_MODE::DENSITY;
}
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}