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

#include "PyramidalSimulation.h"
#include "owPhysicsFluidSimulator.h"

int iter_step = 100;				 // Count of iteration which will be skipped before logging configuration to file
								 // NOTE: this using only in "load config to file" mode

/** Constructor method for owPhysicsFluidSimulator.
 *
 *  @param helper
 *  pointer to owHelper object with helper function.
 *  @param dev_type
 *  defines preferable device type for current configuration
 */
owPhysicsFluidSimulator::owPhysicsFluidSimulator(owHelper * helper,int argc, char ** argv)
{
	//int generateInitialConfiguration = 1;//1 to generate initial configuration, 0 - load from file

	try{
		iterationCount = 0;
		config = new owConfigProrerty(argc, argv);
		// LOAD FROM FILE
		owHelper::preLoadConfiguration(config);

		//TODO move initialization to configuration class
		config->gridCellsX = (int)( ( config->xmax - config->xmin ) / h ) + 1;
		config->gridCellsY = (int)( ( config->ymax - config->ymin ) / h ) + 1;
		config->gridCellsZ = (int)( ( config->zmax - config->zmin ) / h ) + 1;
		config->gridCellCount = config->gridCellsX * config->gridCellsY * config->gridCellsZ;
		//
		position_cpp = new float[ 4 * config->getParticleCount() ];
		velocity_cpp = new float[ 4 * config->getParticleCount() ];
		acceleration_cpp = new float[ config->getParticleCount() * 3 * 4 ];
		normales_cpp = new float[ 4 * config->getParticleCount()];
		muscle_activation_signal_cpp = new float [config->MUSCLE_COUNT];
		if(config->numOfElasticP != 0)
			elasticConnectionsData_cpp = new float[ 4 * config->numOfElasticP * MAX_NEIGHBOR_COUNT ];
		if(config->numOfMembranes<=0)
			membraneData_cpp = NULL;
		else
			membraneData_cpp = new int [config->numOfMembranes*3];
		if(config->numOfElasticP<=0)
			particleMembranesList_cpp = NULL;
		else
			particleMembranesList_cpp = new int [config->numOfElasticP * MAX_MEMBRANES_INCLUDING_SAME_PARTICLE];
		for(int i=0;i<config->MUSCLE_COUNT;i++)
		{
			muscle_activation_signal_cpp[i] = 0.f;
		}

		//The buffers listed below are only for usability and debug
		density_cpp = new float[ 1 * config->getParticleCount() ];
		particleIndex_cpp = new unsigned int[config->getParticleCount() * 2];

		// LOAD FROM FILE
		owHelper::loadConfiguration( position_cpp, velocity_cpp, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp, config );		//Load configuration from file to buffer

		if(config->numOfElasticP != 0){
			ocl_solver = new owOpenCLSolver(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp);	//Create new openCLsolver instance
		}else
			ocl_solver = new owOpenCLSolver(position_cpp,velocity_cpp, config);	//Create new openCLsolver instance
		this->helper = helper;
	}catch(std::runtime_error &re){
		std::cout << "ERROR: " << re.what() << std::endl;
		exit( -1 );
	}
	catch( std::exception &e ){
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
	// Free all buffers
	cleanBuffers();
	config->resetNeuronSimulation();
	iterationCount = 0;
	config->numOfBoundaryP = 0;
	config->numOfElasticP = 0;
	config->numOfLiquidP = 0;
	config->numOfMembranes = 0;
	// LOAD FROM FILE
	owHelper::preLoadConfiguration(config);
	//TODO move initialization to configuration class
	config->gridCellsX = (int)( ( config->xmax - config->xmin ) / h ) + 1;
	config->gridCellsY = (int)( ( config->ymax - config->ymin ) / h ) + 1;
	config->gridCellsZ = (int)( ( config->zmax - config->zmin ) / h ) + 1;
	config->gridCellCount = config->gridCellsX * config->gridCellsY * config->gridCellsZ;
	//
	position_cpp = new float[ 4 * config->getParticleCount() ];
	velocity_cpp = new float[ 4 * config->getParticleCount() ];

	muscle_activation_signal_cpp = new float [config->MUSCLE_COUNT];
	if(config->numOfElasticP != 0) elasticConnectionsData_cpp = new float[ 4 * config->numOfElasticP * MAX_NEIGHBOR_COUNT ];
	if(config->numOfMembranes<=0) membraneData_cpp = NULL; else membraneData_cpp = new int [ config->numOfMembranes * 3 ];
	if(config->numOfElasticP<=0)  particleMembranesList_cpp = NULL; else particleMembranesList_cpp = new int [config->numOfElasticP*MAX_MEMBRANES_INCLUDING_SAME_PARTICLE];
	for(int i=0;i<config->MUSCLE_COUNT;i++)
	{
		muscle_activation_signal_cpp[i] = 0.f;
	}

	//The buffers listed below are only for usability and debug
	density_cpp = new float[ 1 * config->getParticleCount() ];
	particleIndex_cpp = new unsigned int[config->getParticleCount() * 2];

	// LOAD FROM FILE
	owHelper::loadConfiguration( position_cpp, velocity_cpp, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp, config );		//Load configuration from file to buffer
	if(config->numOfElasticP != 0){
		ocl_solver->reset(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp);	//Create new openCLsolver instance
	}else
		ocl_solver->reset(position_cpp,velocity_cpp, config);	//Create new openCLsolver instance
}

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

	//if(iterationCount==1) return 0.0;//uncomment this line to stop movement of the scene

	helper->refreshTime();
	printf("\n[[ Step %d ]]\n",iterationCount);
	try{
		//SEARCH FOR NEIGHBOURS PART
		//ocl_solver->_runClearBuffers();								helper->watch_report("_runClearBuffers: \t%9.3f ms\n");
		ocl_solver->_runHashParticles(config);							helper->watch_report("_runHashParticles: \t%9.3f ms\n");
		ocl_solver->_runSort(config);									helper->watch_report("_runSort: \t\t%9.3f ms\n");
		ocl_solver->_runSortPostPass(config);							helper->watch_report("_runSortPostPass: \t%9.3f ms\n");
		ocl_solver->_runIndexx(config);									helper->watch_report("_runIndexx: \t\t%9.3f ms\n");
		ocl_solver->_runIndexPostPass(config);							helper->watch_report("_runIndexPostPass: \t%9.3f ms\n");
		ocl_solver->_runFindNeighbors(config);							helper->watch_report("_runFindNeighbors: \t%9.3f ms\n");
		//PCISPH PART
		if(config->getIntegrationMethod() == LEAPFROG){ // in this case we should remmember value of position on stem i - 1
			//Calc next time (t+dt) positions x(t+dt)
			ocl_solver->_run_pcisph_integrate(iterationCount,0/*=positions_mode*/, config);
		}
		ocl_solver->_run_pcisph_computeDensity(config);
		ocl_solver->_run_pcisph_computeForcesAndInitPressure(config);
		ocl_solver->_run_pcisph_computeNormales(config);
		ocl_solver->_run_pcisph_calcSurfaceTension(config);
		ocl_solver->_run_pcisph_computeElasticForces(config);
		do{
			//printf("\n^^^^ iter %d ^^^^\n",iter);
			ocl_solver->_run_pcisph_predictPositions(config);
			ocl_solver->_run_pcisph_predictDensity(config);
			ocl_solver->_run_pcisph_correctPressure(config);
			ocl_solver->_run_pcisph_computePressureForceAcceleration(config);
			iter++;
		}while( iter < maxIteration );

		//and finally calculate v(t+dt)
		if(config->getIntegrationMethod() == LEAPFROG){
			ocl_solver->_run_pcisph_integrate(iterationCount,1/*=velocities_mode*/, config);		helper->watch_report("_runPCISPH: \t\t%9.3f ms\t3 iteration(s)\n");
		}
		else{
			ocl_solver->_run_pcisph_integrate(iterationCount, 2,config);		helper->watch_report("_runPCISPH: \t\t%9.3f ms\t3 iteration(s)\n");
		}
		//Handling of Interaction with membranes
		if(config->numOfMembranes > 0){
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
		printf("_Total_step_time:\t%9.3f ms\n",helper->getElapsedTime());
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
		/*if(iterationCount == 150 || iterationCount == 0){
			float low;
			float high;
			float left, right;

			for(int i=0;i<config->getParticleCount();i++){
				if((int)position_cpp[4 * i + 3] == LIQUID_PARTICLE){
					float y = position_cpp[4 * i + 1];
					float x = position_cpp[4 * i + 0];
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
				float volume, density;
				getDensityDistrib();
				if(iterationCount == 0){
					volume = dist_x * dist_x * dist_y * pow(simulationScale,3.0f);
					density = mass * config->numOfLiquidP / volume;
				}else{
					volume = pow(dist_x * simulationScale,3.0f)/6.f;
					density = mass * config->numOfLiquidP / volume;
				}
				outFile << "Volume of water is:" << volume << "\n";
				outFile << "Density of water is:" << density << "\n";
				//outFile << "Initial Volume of cube is:" << start_volume << "\n";
				//outFile << "Initial Density of cube is:" << start_density << "\n";
				outFile << "Simulation Scale is:" << simulationScale << "\n";
				outFile.close();
				if(iterationCount == 150)
					exit(0);
			}
		}*/

		iterationCount++;

        config->updatePyramidalSimulation(muscle_activation_signal_cpp);
		ocl_solver->updateMuscleActivityData(muscle_activation_signal_cpp, config);
		return helper->getElapsedTime();
	}
	catch(std::exception &e)
	{
		std::cout << "ERROR: " << e.what() << std::endl;
		exit( -1 );
	}
}
/** Prepare data and log it to special configuration
 *  file you can run your simulation from place you snapshoted it
 *
 *  @param fileName - name of file where saved configuration will be stored
 */
void owPhysicsFluidSimulator::makeSnapshot(){
	getvelocity_cpp();
	std::string fileName = config->getSnapshotFileName();
	owHelper::loadConfigurationToFile(position_cpp, velocity_cpp, elasticConnectionsData_cpp, membraneData_cpp, particleMembranesList_cpp, fileName.c_str(), config);
}

//Destructor
owPhysicsFluidSimulator::~owPhysicsFluidSimulator(void)
{
	cleanBuffers();
	delete config;
	delete ocl_solver;
}

void owPhysicsFluidSimulator::cleanBuffers(){
	delete [] position_cpp;
	delete [] velocity_cpp;
	delete [] density_cpp;
	delete [] particleIndex_cpp;
	if(config->numOfElasticP != 0)
		delete [] elasticConnectionsData_cpp;
	delete [] muscle_activation_signal_cpp;
	if(membraneData_cpp != NULL) {
		delete [] membraneData_cpp;
		delete [] particleMembranesList_cpp;
	}
}
