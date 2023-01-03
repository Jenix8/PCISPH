#pragma once
#include <vector>
#include <glm/glm.hpp>

using namespace std;
using namespace glm;

class NS
{

	vector<vec3> positions;

private:
	unsigned int Hash(vec3 position)
	{
		unsigned int cell_idx = 0;
		return cell_idx;
	}

public:

	NS()
	{
		unsigned int numCells = 1;
		unsigned int numParticles = 1;

		vector<vector<unsigned int>> particle_idx_in_cell(numCells);

		//clear cells
		for (unsigned int cidx = 0u; cidx < numCells; cidx++)
		{
			// hash = cell idx 
			particle_idx_in_cell[cidx].clear();

		}


		for(unsigned int pidx = 0u; pidx < numParticles; pidx++)
		{
			// hash = cell idx 
			unsigned int cIdx = Hash(positions[pidx]);
			particle_idx_in_cell[cIdx].push_back(pidx);
		}


		for (unsigned int pidx = 0u; pidx < numParticles; pidx++)
		{
			unsigned int cIdx = Hash(positions[pidx]);

			vec3 p_Pos = positions[pidx];
			// for all neighbours 
			unsigned int numNeighbours = particle_idx_in_cell[cIdx].size();
			for (unsigned int idx = 0u; idx < numNeighbours; idx++)
			{
				unsigned int nIdx = particle_idx_in_cell[cIdx][idx];

				vec3 n_Pos = positions[nIdx];

				// do things with relative pos

				vec3 rel_Pos = p_Pos - n_Pos;

				// W (rel_Pos)  
			}
		}
		// particle indices in a cell 


	
	
	};

};