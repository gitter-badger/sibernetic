/*******************************************************************************
 * The MIT License (MIT)
 *
 * Copyright (c) 2011, 2013 OpenWorm.
 * http://openworm.org
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License
 * which accompanies this distribution, and is available at
 * http://opensource.org/licenses/MIT
 *
 * Contributors:
 *     	OpenWorm - http://openworm.org/people.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

#include "owPhysicsFluidSimulator.h"

float calcDelta();
float calcDelta_new();
extern const float delta = calcDelta_new();
int numOfElasticConnections = 0; // Number of elastic connection TODO: move this to owConfig class
int numOfLiquidP = 0;			 // Number of liquid particles TODO: move this to owConfig class
int numOfElasticP = 0;			 // Number of liquid particles TODO: move this to owConfig class
int numOfBoundaryP = 0;			 // Number of boundary particles TODO: move this to owConfig class
int numOfMembranes = 0;			 // Number of membranes TODO: move this to owConfig class
extern float * muscle_activation_signal_cpp;
int iter_step = 10;				 // Count of iteration which will be skipped before logging configuration to file
								 // NOTE: this using only in "load config to file" mode
float diameter = 0;
int nearestParticle = -1;
/** Constructor method for owPhysicsFluidSimulator.
 *
 *  @param helper
 *  pointer to owHelper object with helper function.
 *  @param dev_type
 *  defines preferable device type for current configuration
 */
owPhysicsFluidSimulator::owPhysicsFluidSimulator(owHelper * helper,DEVICE dev_type)
{
	//int generateInitialConfiguration = 1;//1 to generate initial configuration, 0 - load from file

	try{
		iterationCount = 0;
		config = new owConfigProrerty();
		config->setDeviceType(dev_type);
		// LOAD FROM FILE
		//owHelper::preLoadConfiguration(numOfMembranes, config, numOfLiquidP, numOfElasticP, numOfBoundaryP);
		owHelper::generateConfiguration(0,position_cpp,velocity_cpp, numOfLiquidP, numOfElasticP,numOfBoundaryP, config);

		//TODO move initialization to configuration class
		config->gridCellsX = (int)( ( config->xmax - config->xmin ) / h ) + 1;
		config->gridCellsY = (int)( ( config->ymax - config->ymin ) / h ) + 1;
		config->gridCellsZ = (int)( ( config->zmax - config->zmin ) / h ) + 1;
		config->gridCellCount = config->gridCellsX * config->gridCellsY * config->gridCellsZ;
		//
		position_cpp = new float[ 4 * config->getParticleCount() ];
		velocity_cpp = new float[ 4 * config->getParticleCount() ];

		neighbourMap_cpp = new float[MAX_NEIGHBOR_COUNT * 2 * config->getParticleCount()];
		numOfMembranes = 0;
		muscle_activation_signal_cpp = new float [MUSCLE_COUNT];
		if(numOfMembranes<=0)
			membraneData_cpp = NULL;
		else
			membraneData_cpp = new int [numOfMembranes*3];
		if(numOfElasticP<=0)
			particleMembranesList_cpp = NULL;
		else
			particleMembranesList_cpp = new int [numOfElasticP*MAX_MEMBRANES_INCLUDING_SAME_PARTICLE];
		for(int i=0;i<MUSCLE_COUNT;i++)
		{
			muscle_activation_signal_cpp[i] = 0.f;
		}

		//The buffers listed below are only for usability and debug
		density_cpp = new float[ 1 * config->getParticleCount() ];
		particleIndex_cpp = new unsigned int[config->getParticleCount() * 2];
		particleIndex_back_cpp = new unsigned int[config->getParticleCount()];
		// LOAD FROM FILE	
		//owHelper::loadConfiguration( position_cpp, velocity_cpp, elasticConnectionsData_cpp, numOfLiquidP, numOfElasticP, numOfBoundaryP, numOfElasticConnections, numOfMembranes,membraneData_cpp, particleMembranesList_cpp, config );		//Load configuration from file to buffer
		owHelper::generateConfiguration(1,position_cpp,velocity_cpp, numOfLiquidP, numOfElasticP,numOfBoundaryP, config);
		if(numOfElasticP != 0){
			ocl_solver = new owOpenCLSolver(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp);	//Create new openCLsolver instance
		}else
			ocl_solver = new owOpenCLSolver(position_cpp,velocity_cpp, config);	//Create new openCLsolver instance
		this->helper = helper;
	}catch( std::exception &e ){
		std::cout << "ERROR: " << e.what() << std::endl;
		exit( -1 );
	}
}
/** Reset simulation
 *
 *  Restart simulation with new or current simulation configuration.
 *  It redefines all required data buffers and restart owOpenCLSolver
 *  by run owOpenCLSolver::reset(...).
 */
void owPhysicsFluidSimulator::reset(){
	iterationCount = 0;
	numOfBoundaryP = 0;
	numOfElasticP = 0;
	numOfLiquidP = 0;
	numOfMembranes = 0;					

	numOfElasticConnections = 0;
	// LOAD FROM FILE
	owHelper::preLoadConfiguration(numOfMembranes, config, numOfLiquidP, numOfElasticP, numOfBoundaryP);
	//TODO move initialization to configuration class
	config->gridCellsX = (int)( ( config->xmax - config->xmin ) / h ) + 1;
	config->gridCellsY = (int)( ( config->ymax - config->ymin ) / h ) + 1;
	config->gridCellsZ = (int)( ( config->zmax - config->zmin ) / h ) + 1;
	config->gridCellCount = config->gridCellsX * config->gridCellsY * config->gridCellsZ;
	//
	position_cpp = new float[ 4 * config->getParticleCount() ];
	velocity_cpp = new float[ 4 * config->getParticleCount() ];

	muscle_activation_signal_cpp = new float [MUSCLE_COUNT];
	if(numOfMembranes<=0) membraneData_cpp = NULL; else membraneData_cpp = new int [numOfMembranes*3];
	if(numOfElasticP<=0)  particleMembranesList_cpp = NULL; else particleMembranesList_cpp = new int [numOfElasticP*MAX_MEMBRANES_INCLUDING_SAME_PARTICLE];
	for(int i=0;i<MUSCLE_COUNT;i++)
	{
		muscle_activation_signal_cpp[i] = 0.f;
	}

	//The buffers listed below are only for usability and debug
	density_cpp = new float[ 1 * config->getParticleCount() ];
	particleIndex_cpp = new unsigned int[config->getParticleCount() * 2];

	// LOAD FROM FILE
	owHelper::loadConfiguration( position_cpp, velocity_cpp, elasticConnectionsData_cpp, numOfLiquidP, numOfElasticP, numOfBoundaryP, numOfElasticConnections, numOfMembranes,membraneData_cpp, particleMembranesList_cpp, config );		//Load configuration from file to buffer
	if(numOfElasticP != 0){
		ocl_solver->reset(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp);	//Create new openCLsolver instance
	}else
		ocl_solver->reset(position_cpp,velocity_cpp, config);	//Create new openCLsolver instance
}
//TODO delete this after FIX

void owPhysicsFluidSimulator::getDensityDistrib(){
	float minDistVal = 0.0f;
	float maxDistVal = 1100.0f;
	float step = 50.0f;
	const int size = (int)((maxDistVal - minDistVal)/step);
	std::vector<int> distrib(2 * size,0);
	int pib;
	float rho;
	this->getDensity_cpp();
	this->getParticleIndex_cpp();
	for(int i=0;i<config->getParticleCount();i++)
	{
		pib = particleIndex_cpp[2*i + 1];
		particleIndex_cpp[2*pib + 0] = i;
	}
	for(int i=0;i < size;i++){
		distrib[i*2 + 0] = minDistVal + i * step;
		distrib[i*2 + 1] = 0;
	}
	for(int i=0;i<config->getParticleCount();i++){
		if((int)position_cpp[4 * i + 3] == LIQUID_PARTICLE){
			rho = density_cpp[ particleIndex_cpp[ i * 2 + 0 ] ];
			if(rho >= minDistVal && rho <= maxDistVal){
				int index = (int)((rho - minDistVal)/step);
				distrib[2 * index + 1] += 1;
			}else{
				if(rho < minDistVal)
					distrib[0 + 1] += 1;
				else
					distrib[size - 1] += 1;
			}
		}
	}
	std::stringstream ss;
	ss << iterationCount;
	std::string fileName = "./logs/density_distrib_" + ss.str()+".txt";
	owHelper::log_buffer(&distrib[0], 2, size, fileName.c_str());
}
float start_density = 0.f;
float start_volume = 0.f;
/** Run one simulation step
 *
 *  Run simulation step in pipeline manner.
 *  It starts with neighbor search algorithm than
 *  physic simulation algorithms: PCI SPH [1],
 *  elastic matter simulation, boundary handling [2],
 *  membranes handling and finally numerical integration.
 *  [1] http://www.ifi.uzh.ch/vmml/publications/pcisph/pcisph.pdf
 *  [2] M. Ihmsen, N. Akinci, M. Gissler, M. Teschner,
 *      Boundary Handling and Adaptive Time-stepping for PCISPH
 *      Proc. VRIPHYS, Copenhagen, Denmark, pp. 79-88, Nov 11-12, 2010
 *
 *  @param looad_to
 *  If it's true than Sibernetic works "load simulation data in file" mode.
 */
double owPhysicsFluidSimulator::simulationStep(const bool load_to)
{
	int iter = 0;//PCISPH prediction-correction iterations counter
                 //
	// now we will implement sensory system of the c. elegans worm, mechanosensory one
	// here we plan to implement the part of openworm sensory system, which is still
	// one of the grand challenges of this project

	//if(iterationCount==0) return 0.0;//uncomment this line to stop movement of the scene
	//if(iterationCount>100) exit(0);//uncomment this line to stop movement of the scene

	helper->refreshTime();
	printf("\n[[ Step %d ]]\n",iterationCount);
	try{
		//SEARCH FOR NEIGHBOURS PART
		ocl_solver->_runClearBuffers(config);								helper->watch_report("_runClearBuffers: \t%9.3f ms\n");
		ocl_solver->_runHashParticles(config);							helper->watch_report("_runHashParticles: \t%9.3f ms\n");
		ocl_solver->_runSort(config);									helper->watch_report("_runSort: \t\t%9.3f ms\n");
		ocl_solver->_runSortPostPass(config);							helper->watch_report("_runSortPostPass: \t%9.3f ms\n");
		ocl_solver->_runIndexx(config);									helper->watch_report("_runIndexx: \t\t%9.3f ms\n");
		ocl_solver->_runIndexPostPass(config);							helper->watch_report("_runIndexPostPass: \t%9.3f ms\n");
		ocl_solver->_runFindNeighbors(config);							helper->watch_report("_runFindNeighbors: \t%9.3f ms\n");
		//PCISPH PART
		ocl_solver->_run_pcisph_computeDensity(config);
		ocl_solver->_run_pcisph_computeForcesAndInitPressure(config);
		ocl_solver->_run_pcisph_computeElasticForces(config);
		do{
			//printf("\n^^^^ iter %d ^^^^\n",iter);
			ocl_solver->_run_pcisph_predictPositions(config);
			ocl_solver->_run_pcisph_predictDensity(config);
			ocl_solver->_run_pcisph_correctPressure(config);
			ocl_solver->_run_pcisph_computePressureForceAcceleration(config);
			iter++;
		}while( iter < maxIteration );

		ocl_solver->_run_pcisph_integrate(iterationCount, config);		helper->watch_report("_runPCISPH: \t\t%9.3f ms\t3 iteration(s)\n");
		//Handling of Interaction with membranes
		if(numOfMembranes > 0){
			ocl_solver->_run_clearMembraneBuffers(config);
			ocl_solver->_run_computeInteractionWithMembranes(config);
			// compute change of coordinates due to interactions with membranes
			ocl_solver->_run_computeInteractionWithMembranes_finalize(config);
																		helper->watch_report("membraneHadling: \t%9.3f ms\n");
		}
		//END
		ocl_solver->read_position_buffer(position_cpp, config);				helper->watch_report("_readBuffer: \t\t%9.3f ms\n");

		//END PCISPH algorithm
		printf("------------------------------------\n");
		printf("_Total_step_time:\t%9.3f ms\n",helper->get_elapsedTime());
		printf("------------------------------------\n");
		if(load_to){
			if(iterationCount == 0){
				owHelper::loadConfigurationToFile(position_cpp,  config,elasticConnectionsData_cpp,membraneData_cpp,true);
			}else{
				if(iterationCount % iter_step == 0){
					owHelper::loadConfigurationToFile(position_cpp, config, NULL, NULL, false);
				}
			}
		}
		/*TODO delete block below after fix*/
		//Calculating values of volume and density for start and end configuration
		if(iterationCount == 4300 || iterationCount == 0){
			float low;
			float high;
			float left, right;

			for(int i=0;i<config->getParticleCount();i++){
				if((int)position_cpp[4 * i + 3] == LIQUID_PARTICLE){
					float y = position_cpp[4 * i + 1]/* - config->xmax/2*/;
					float x = position_cpp[4 * i + 0]/* - config->xmax/2*/;
					if(i == 0){
						low = y;
						high = y;
						left = x;
						right = x;
					}
					else{
						if(y<low)
							low = y;
						if(y>high)
							high = y;
						if(x<left)
							left = x;
						if(x>right)
							right = x;
					}
				}
			}
			
			if(high - low != 0.0f && left - right != 0.0f){
				float dist_x = fabs(left - right);
				float dist_y = fabs(high - low);
				std::string filename = (iterationCount == 0)?"./logs/Output_Parametrs_0.txt":"./logs/Output_Parametrs.txt";
				std::ofstream outFile (filename.c_str());
				//if(iterationCount % 1000 == 0)
				getDensityDistrib(); //Load histogram of distribution of density to file
				float volume = dist_x * dist_x * dist_y * pow(simulationScale,3.0f);
				float density = mass * numOfLiquidP / volume;
				outFile << "Volume of water is:" << volume << "\n";
				outFile << "Density of water is:" << density << "\n";
				//outFile << "Initial Volume of cube is:" << start_volume << "\n";
				//outFile << "Initial Density of cube is:" << start_density << "\n";
				outFile << "Simulation Scale is:" << simulationScale << "\n";
				centerMass[0] = 0.f;
				centerMass[1] = 0.f;
				centerMass[2] = 0.f;
				for(int i=0;i < config->getParticleCount();i++){
					if((int)position_cpp[4*i+3]==LIQUID_PARTICLE){
						centerMass[0] += position_cpp[4*i + 0];
						centerMass[1] += position_cpp[4*i + 1];
						centerMass[2] += position_cpp[4*i + 2];
					}
				}
				centerMass[0] /= numOfLiquidP;
				centerMass[1] /= numOfLiquidP;
				centerMass[2] /= numOfLiquidP;
				float dist = -1.0f;
				float dist_temp;
				for(int i=0;i<config->getParticleCount();i++){
					if((int)position_cpp[4*i+3]==LIQUID_PARTICLE){
						float x = centerMass[0] - position_cpp[4 * i + 0];
						float y = centerMass[1] - position_cpp[4 * i + 1];
						float z = centerMass[2] - position_cpp[4 * i + 2];
						dist_temp = (x * x + y * y + z * z);
						if(dist == -1.0f){
							dist = dist_temp;
							nearestParticle = i;
						}else if(dist > dist_temp){
							dist = dist_temp;
							nearestParticle = i;
						}
					}
				}
				outFile << "NearestParticle particle is:" << nearestParticle << "\n";
				getDensity_cpp();
				int pib;
				getParticleIndexBack_cpp();
				
				float * correctDensity = new float[config->getParticleCount() * 1];
				for(int i=0; i<config->getParticleCount() ;i++)
					correctDensity[i] = density_cpp[ particleIndex_back_cpp[i] ];
				if(nearestParticle > -1)
					outFile << "Density of nearestParticle particle is:" << correctDensity[nearestParticle] << "\n";
				outFile.close();
				float * dist_dens_distr = new float[ 2 * numOfLiquidP ];
				for(int i=0;i<config->getParticleCount();i++){
					if((int)position_cpp[4*i+3]==LIQUID_PARTICLE){
						dist_dens_distr[i*2 + 0] = position_cpp[ i * 4 + 1];
						dist_dens_distr[i*2 + 1] = correctDensity[i];
					}
				}
				owHelper::log_buffer(dist_dens_distr,2, numOfLiquidP,"./logs/density_13000.txt");
				getNeghboursMap();
				int startIndex = particleIndex_back_cpp[nearestParticle] * MAX_NEIGHBOR_COUNT;
				int endIndex = startIndex + MAX_NEIGHBOR_COUNT; 
				float * neighborCorrect = new float[numOfLiquidP*MAX_NEIGHBOR_COUNT*2];
				//owHelper::log_buffer(neighbourMap_cpp,2, endIndex,"./logs/neighborMap_for_nearestParticle.txt", startIndex);
				for(int i=0;i < config->getParticleCount();i++){
					if((int)position_cpp[4*i+3] != BOUNDARY_PARTICLE){
						int id = particleIndex_back_cpp[i];
						int s_id = id * MAX_NEIGHBOR_COUNT;
						int k = 0;
						for(int j = i * MAX_NEIGHBOR_COUNT; j < ( i + 1 ) * MAX_NEIGHBOR_COUNT; j++)
						{
							neighborCorrect[2 * j + 0] = neighbourMap_cpp[2 * s_id + 2 * k + 0];
							neighborCorrect[2 * j + 1] = neighbourMap_cpp[2 * s_id + 2 * k + 1];
							k++;
						}
					}
				}
				//owHelper::log_buffer(neighbourMap_cpp,2, numOfLiquidP,"./logs/neighborMap_for_nearestParticle.txt");
				owHelper::log_buffer(neighborCorrect,2, numOfLiquidP,"./logs/neighborMap_for_liquid.txt");
				owHelper::log_buffer(position_cpp,4, numOfLiquidP,"./logs/position_2300.txt");
			}
			//float left;
			//float right;
			//for(int i=0;i<config->getParticleCount();i++){
			//	if((int)position_cpp[4 * i + 3] == LIQUID_PARTICLE){
			//		float x = position_cpp[4 * i + 0]/* - config->xmax/2*/;
			//		if(i == 0){
			//			left = x;
			//			right = x;
			//		}
			//		else{
			//			if(x<left)
			//				left = x;
			//			if(x>right)
			//				right = x;
			//		}
			//	}
			//}
			//if(right - left != 0){
			//	owHelper::log_buffer(position_cpp,4, config->getParticleCount(),"./logs/position.txt");
			//	diameter = fabs(right - left) ;//* simulationScale;
			//	float volume;
			//	float density;
			//	if(iterationCount != 0 && iterationCount % 100 == 0){
			//		std::ofstream outFile ("./logs/Output_Parametrs.txt");
			//		if(iterationCount % 1000 == 0)
			//			getDensityDistrib(); //Load histogram of distribution of density to file
			//		volume = pow(diameter*simulationScale,3.0f) * 3.14159265359f * 1.0f/6.0f;
			//		density = mass * numOfLiquidP / volume;
			//		outFile << "Diameter of droplet is:" << diameter*simulationScale << "\n";
			//		outFile << "Volume of droplet is:" << volume << "\n";
			//		outFile << "Density of droplet is:" << density << "\n";
			//		outFile << "Initial Volume of cube is:" << start_volume << "\n";
			//		outFile << "Initial Density of cube is:" << start_density << "\n";
			//		outFile << "Simulation Scale is:" << simulationScale << "\n";
			//		outFile.close();
			//		for(int i=0;i < config->getParticleCount();i++){
			//			if((int)position_cpp[4*i+3]==LIQUID_PARTICLE){
			//				centerMass[0] += position_cpp[4*i + 0];
			//				centerMass[1] += position_cpp[4*i + 1];
			//				centerMass[2] += position_cpp[4*i + 2];
			//			}
			//		}
			//		centerMass[0] /= numOfLiquidP;
			//		centerMass[1] /= numOfLiquidP;
			//		centerMass[2] /= numOfLiquidP;
			//		/*std::cout << "Diameter of droplet is:" << diameter << std::endl;
			//		std::cout << "Volume of droplet is:" << volume << std::endl;
			//		std::cout << "Density of droplet is:" << density << std::endl;
			//		std::cout << "Initial Volume of cube is:" << start_volume << std::endl;
			//		std::cout << "Initial Density of cube is:" << start_density << std::endl;
			//		std::cout << "Simulation Scale is:" << simulationScale << std::endl;*/
			//		//exit(0);
			//	}
			//	else if(iterationCount == 0){
			//		start_volume = pow(10.0f * r0 * simulationScale,3.0f);//it because on first iteration it's a cube
			//		start_density = mass * numOfLiquidP / start_volume;
			//		getDensityDistrib(); //Load histogram of distribution of density to file

			//	}
			//	//std::cout << "Density of droplet is:" << simulationScale << std::endl;
			//}
			//else{
			//	std::cout << "ERROR"<< std::endl;
			//}
		}
		/*END*/
		iterationCount++;
		//for(int i=0;i<MUSCLE_COUNT;i++) { muscle_activation_signal_cpp[i] *= 0.9f; }
		ocl_solver->updateMuscleActivityData(muscle_activation_signal_cpp);
		return helper->get_elapsedTime();
	}
	catch(std::exception &e)
	{
		std::cout << "ERROR: " << e.what() << std::endl;
		exit( -1 );
	}
}

//Destructor
owPhysicsFluidSimulator::~owPhysicsFluidSimulator(void)
{
	delete [] position_cpp;
	delete [] velocity_cpp;
	delete [] density_cpp;
	delete [] particleIndex_cpp;
	delete [] muscle_activation_signal_cpp;
	if(membraneData_cpp != NULL) delete [] membraneData_cpp;
	delete config;
	delete ocl_solver;
}
/** Calculating delta parameter.
 *
 *  "In these situations,
 *  the SPH equations result in falsified values. To circumvent that problem, we pre-
 *  compute a single scaling factor δ according to the following formula [1, eq. 8] which is
 *  evaluated for a prototype particle with a filled neighborhood. The resulting value
 *  is then used for all particles. Finally, we end up with the following equations
 *  which are used in the PCISPH method" [1].
 *  [1] http://www.ifi.uzh.ch/vmml/publications/pcisph/pcisph.pdf
 */
float calcDelta()
{

	float C0 = 3.f * (sqrt(5.0f) - 1.f) / 4.f; //= 0.927050983124842272306880251548
	float C1 = 9.f * (9.f + sqrt(5.0f)) / 76.f;//= 1.33058699733550141141687582919
	float C2 = 9.f * (7.f + 5 * sqrt(5.0f)) / 76.f;//= 2.15293498667750705708437914596
	float C3 = 3.f * (1.f + sqrt(5.0f)) / 4.f;     //= 2.427050983124842272306880251548

	/*V0  = ( 0.0,   C0,   C3)
	V1  = ( 0.0,   C0,  -C3)
	V2  = ( 0.0,  -C0,   C3)
	V3  = ( 0.0,  -C0,  -C3)
	V4  = (  C3,  0.0,   C0)
	V5  = (  C3,  0.0,  -C0)
	V6  = ( -C3,  0.0,   C0)
	V7  = ( -C3,  0.0,  -C0)
	V8  = (  C0,   C3,  0.0)
	V9  = (  C0,  -C3,  0.0)
	V10 = ( -C0,   C3,  0.0)
	V11 = ( -C0,  -C3,  0.0)
	V12 = (  C1,  0.0,   C2)
	V13 = (  C1,  0.0,  -C2)
	V14 = ( -C1,  0.0,   C2)
	V15 = ( -C1,  0.0,  -C2)
	V16 = (  C2,   C1,  0.0)
	V17 = (  C2,  -C1,  0.0)
	V18 = ( -C2,   C1,  0.0)
	V19 = ( -C2,  -C1,  0.0)
	V20 = ( 0.0,   C2,   C1)
	V21 = ( 0.0,   C2,  -C1)
	V22 = ( 0.0,  -C2,   C1)
	V23 = ( 0.0,  -C2,  -C1)
	V24 = ( 1.5,  1.5,  1.5)
	V25 = ( 1.5,  1.5, -1.5)
	V26 = ( 1.5, -1.5,  1.5)
	V27 = ( 1.5, -1.5, -1.5)
	V28 = (-1.5,  1.5,  1.5)
	V29 = (-1.5,  1.5, -1.5)
	V30 = (-1.5, -1.5,  1.5)
	V31 = (-1.5, -1.5, -1.5)*/

	//float x[] = { 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 2,-2, 0, 0, 0, 0, 0, 0 };
    //float y[] = { 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 0, 2,-2, 0, 0, 0, 0 };
    //float z[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 2,-2, 1,-1 };
	float x[] = { 0,   0,   0,   0, C3, C3,-C3, -C3, C0,  C0, -C0, -C0, C1,  C1, -C1, -C1, C2,  C2, -C2, -C2,  0,   0,   0,   0, 1.5,  1.5,  1.5,  1.5, -1.5, -1.5, -1.5, -1.5 };
	float y[] = { C0, C0, -C0, -C0,  0, 0,  0,    0, C3, -C3,  C3, -C3,  0,   0,   0,   0, C1, -C1,  C1, -C1, C2,  C2, -C2, -C2, 1.5,  1.5, -1.5, -1.5,  1.5,  1.5, -1.5, -1.5 };
	float z[] = { C3,-C3,  C3, -C3, C0,-C0, C0, -C0,  0,   0,   0,   0, C2, -C2,  C2, -C2,  0,   0,   0,   0, C1, -C1,  C1, -C1, 1.5, -1.5,  1.5, -1.5,  1.5, -1.5,  1.5, -1.5 };
	float x_c=0.f, y_c=0.f,z_c=0.f;
	for(int i =0;i<32;i++)
	{
		float dist1 = sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);//scaled, right?
		x[i] /= dist1;
		y[i] /= dist1;
		z[i] /= dist1;
	}
	float sum1_x = 0.f;
	float sum1_y = 0.f;
	float sum1_z = 0.f;
    double sum1 = 0.0, sum2 = 0.0;
	float v_x = 0.f;
	float v_y = 0.f;
	float v_z = 0.f;
	float dist;
	double density = 0;
	float hScaled2 = (h * simulationScale) * (h * simulationScale);
	float r_ij2;
	/* I suppose that particle radius here should equal to real length of r0 = h/2 (real length r0 * simulationScale)
	 * because we generate configuration and arrange particles such way that they located from each other not further that r0
	 */
	float particleRadius = simulationScale * r0 ;//* 1.13f;/*pow(mass/rho0,1.f/3.f);*/  //

	float h_r_2;
	int counter = 0;
    for (int i = 0; i < MAX_NEIGHBOR_COUNT; i++)
    {
		v_x = x[i] * particleRadius;
		v_y = y[i] * particleRadius;
		v_z = z[i] * particleRadius;

        dist = sqrt(v_x*v_x+v_y*v_y+v_z*v_z);//scaled, right?

        if (dist <= h*simulationScale)
        {
			h_r_2 = pow((h*simulationScale - dist),2);//scaled

            sum1_x += h_r_2 * v_x / dist;
			sum1_y += h_r_2 * v_y / dist;
			sum1_z += h_r_2 * v_z / dist;
			counter++;
            sum2 += h_r_2 * h_r_2;
            r_ij2 = dist * dist;
            density += (hScaled2-r_ij2)*(hScaled2-r_ij2)*(hScaled2-r_ij2);//TODO delete this after fix
        }
    }
    density *= mass_mult_Wpoly6Coefficient;
	sum1 = sum1_x*sum1_x + sum1_y*sum1_y + sum1_z*sum1_z;
	double result = 1.0 / (beta * gradWspikyCoefficient * gradWspikyCoefficient * (sum1 + sum2));
	//return  1.0f / (beta * gradWspikyCoefficient * gradWspikyCoefficient * (sum1 + sum2));
	return (float)result;
}

float calcDelta_new()
{
	float d[] = { 4.13207e-006, 4.66567e-006, 4.4566e-006, 3.93635e-006, 4.42284e-006, 4.09012e-006, 5.28614e-006, \
		5.87074e-006, 2.68782e-006, 3.73287e-006, 4.93145e-006, 5.71927e-006, 4.3676e-006, 5.44214e-006, 3.22728e-006, 4.56464e-006, \
		5.99892e-006, 1.89576e-006, 5.85675e-006, 4.52719e-006, 4.28481e-006, 4.78682e-006, 5.21755e-006, 3.89801e-006, 5.9193e-006, \
		5.9486e-006, 3.4955e-006, 4.21114e-006, 5.51424e-006, 5.92427e-006, 4.18299e-006, 4.52461e-006};
	float sum1_x = 0.f;
	float sum1_y = 0.f;
	float sum1_z = 0.f;
    double sum1 = 0.0, sum2 = 0.0;
	float v_x = 0.f;
	float v_y = 0.f;
	float v_z = 0.f;
	float dist;
	double density = 0;
	float hScaled2 = (h * simulationScale) * (h * simulationScale);
	float r_ij2;
	/* I suppose that particle radius here should equal to real length of r0 = h/2 (real length r0 * simulationScale)
	 * because we generate configuration and arrange particles such way that they located from each other not further that r0
	 */
	float particleRadius = simulationScale * r0 ;//* 1.13f;/*pow(mass/rho0,1.f/3.f);*/  //

	float h_r_2;
	int counter = 0;
    for (int i = 0; i < MAX_NEIGHBOR_COUNT; i++)
    {
        dist = d[i];

        if (dist <= h*simulationScale)
        {
			h_r_2 = pow((h*simulationScale - dist),2);//scaled

            sum1_x += h_r_2 * v_x / dist;
			sum1_y += h_r_2 * v_y / dist;
			sum1_z += h_r_2 * v_z / dist;
			counter++;
            sum2 += h_r_2 * h_r_2;
            r_ij2 = dist * dist;
            density += (hScaled2-r_ij2)*(hScaled2-r_ij2)*(hScaled2-r_ij2);//TODO delete this after fix
        }
    }
    density *= mass_mult_Wpoly6Coefficient;
	sum1 = sum1_x*sum1_x + sum1_y*sum1_y + sum1_z*sum1_z;
	double result = 1.0 / (beta * gradWspikyCoefficient * gradWspikyCoefficient * (sum1 + sum2));
	//return  1.0f / (beta * gradWspikyCoefficient * gradWspikyCoefficient * (sum1 + sum2));
	return (float)result;
}