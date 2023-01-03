#include <glad/glad.h> // 이게 먼저
#include <GLFW/glfw3.h> 

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include "simul.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include "CompactNSearch.h"
using namespace CompactNSearch;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// initializing setup
const unsigned int WIDTH = 1280;
const unsigned int HEIGHT = 720;

// Constants
constexpr float PI = 3.1415926535;

// camera setup
Camera camera(glm::vec3(3.0f, 2.0f, 3.0f));
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

// lighting
glm::vec3 lightPos(4.0f, 4.0f, 4.0f);

// particle setting
float sphereVertices[2160];
float Wall = 1.0f;
float h = 0.025f;					// particle radius
float rhoZero = 1000;				// reference density
float m = rhoZero * 4 * PI * h * h * h / 3;	// particle mass (all particles have equal masses)
glm::vec3 g(0.0f, -9.81f, 0.0f);	// gravity
glm::vec3 Fg = m * g;

// particles
std::vector<glm::vec3> x;			// position
std::vector<glm::vec3> v;			// velocity
std::vector<bool> isWat;			// true: water, false: boundary
std::vector<glm::vec3> px;			// predicted position
std::vector<glm::vec3> pv;			// predicted velocity
std::vector<float> d;				// density
std::vector<float> dErr;			// density variation
float delta = 0.003f;
float eta = 0.01f;

// another 
int minIterations = 3;
int maxIterations = 10;
float deltaTime = 1 / 400.f;// 0.0013f;

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

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { 
		std::cout << "Failed to initialize GLAD" << '\n';
		return -1;
	}

	glEnable(GL_DEPTH_TEST);

	Shader sphereShader("color.vs", "color.fs");
	Shader lightCubeShader("light_cube.vs", "light_cube.fs");

	InitSphere(sphereVertices);

	unsigned int sphereVBO, sphereVAO;
	glGenVertexArrays(1, &sphereVAO);
	glGenBuffers(1, &sphereVBO);
	glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sphereVertices), sphereVertices, GL_STATIC_DRAW);
	glBindVertexArray(sphereVAO);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// normal attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);


	//////////////////////////////////////////////////
	// Start Simulation

	initialize(x, isWat);
	int pCount = getParticleCount(isWat);
	int allPtc = x.size();

	std::vector<glm::vec3> initVec3(allPtc);
	std::vector<float> initFlt(allPtc);
	v = initVec3;
	px = x;
	pv = v;
	d = initFlt;
	dErr = initFlt;
	printf("pCount: %d\n", pCount);

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		// 1. DEFAULT SETTING

		// input
		processInput(window);

		glClearColor(0.6f, 0.5f, 0.4f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// 2. ANIMATING
		// Bounding Volume is [-1.5, 1.5] x [0.0, 3.0] x [-1.5, 1.5] 
		
		// find neighborhoods
		std::vector<std::vector<int>> nParticlesList = createNeighborList(x, isWat);

		// initialize pressure and pressure force as 0
		std::vector<float> p(allPtc);			// pressure
		std::vector<glm::vec3> Fp(allPtc);		// pressure Force
		std::vector<glm::vec3> accel(allPtc);	// acceleration

		// START PREDICTING LOOP
		int Iter = 0;
		while ((errCheck(dErr, rhoZero * eta) || Iter < minIterations))// && Iter < maxIterations)
		{
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)
				for (int i = 0; i < allPtc; i++)
				{
					if (!isWat[i]) continue;

					// predict next v and x
					accel[i] = (Fg + Fp[i]) / m;

					pv[i] = v[i] + deltaTime * accel[i];
					px[i] = x[i] + deltaTime * pv[i];
				}

				#pragma omp for schedule(static)
				for (int i = 0; i < allPtc; i++)
				{
					// predict next density and its variation
					if (!isWat[i]) d[i] = rhoZero;
					else d[i] = predictDensity(i, px, nParticlesList);

					dErr[i] = d[i] - rhoZero;

					// update pressure
					p[i] += dErr[i] * delta;
					p[i] = std::max(p[i], 0.f);
				}

				#pragma omp for schedule(static)
				for (int i = 0; i < allPtc; i++)
				{
					if (!isWat[i]) continue;

					// compute Fp
					for (int n = 0; n < nParticlesList[i].size(); n++)
					{
						if (i == nParticlesList[i][n]) continue;
						Fp[i] -= CalcPressForce(i, nParticlesList[i][n], px, p, d, isWat);
					}
				}
			}

			Iter++;
		}
		
		// assign new v and x
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)
			for (int i = 0; i < allPtc; i++)
			{
				if (!isWat[i]) continue;

				v[i] = pv[i];
				x[i] = px[i];
			}
		}

		// 3. RENDERING

		// activate shader
		sphereShader.use();
		sphereShader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
		sphereShader.setVec3("lightPos", lightPos);
		sphereShader.setVec3("viewPos", camera.Position);

		// create transformations
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera.GetViewMatrix();
		sphereShader.setMat4("projection", projection);
		sphereShader.setMat4("view", view);

		for (int i = 0; i < allPtc; i++)
		{
			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model, x[i]);
			sphereShader.setMat4("model", model);

			if (isWat[i]) sphereShader.setVec3("objectColor", 0.11f, 0.64f, 0.96f);
			else sphereShader.setVec3("objectColor", 0.2f, 0.2f, 0.2f);

			if (isWat[i] || (x[i][0] <= Wall && x[i][2] <= Wall))
			{
				glBindVertexArray(sphereVAO);
				glDrawArrays(GL_TRIANGLES, 0, 360);
			}
		}

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// optional: de-allocate all resources once they've outlived their purpose:
	glDeleteVertexArrays(1, &sphereVAO);
	glDeleteBuffers(1, &sphereVBO);

	// glfw: terminate, clearing all previously allocated GLFW resources.
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
	float T = 100 * deltaTime;
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