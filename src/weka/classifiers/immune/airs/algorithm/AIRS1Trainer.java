/*
 * Created on 30/12/2004
 * Last Edited on 21/2/2012
 * map/reduce version
 */
package weka.classifiers.immune.airs.algorithm;


import global.Global;
import java.io.BufferedReader;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.text.NumberFormat;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

import weka.classifiers.immune.airs.algorithm.classification.MajorityVote;
import weka.classifiers.immune.airs.algorithm.initialisation.RandomInstancesInitialisation;
import weka.classifiers.immune.airs.algorithm.samplegeneration.RandomMutate;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Type: AIRS1Trainer
 * File: AIRS1Trainer.java
 * Date: 30/12/2004
 * Last Edit: 21/2/2012
 *
 * Description:
 *
 * @author Jason Brownlee
 * @Edited Neda Soltani
 *
 */
public class AIRS1Trainer implements AISTrainer
{
	protected final double affinityThresholdScalar;
	protected final double clonalRate;
	protected final double hyperMutationRate;
	protected final double mutationRate;
	protected final double totalResources;
	protected final double stimulationThreshold;
	protected final int affinityThresholdNumInstances;
	protected final Random rand;
	protected final int arbCellPoolInitialSize;
	protected final int memoryCellPoolInitialSize;
	protected final int kNN;


	//added by neda
	protected final int Dist;
	protected final boolean Update;
	protected final boolean NegativeSel;
	protected final boolean UserSpec;



	protected AffinityFunction affinityFunction;
	protected SampleGenerator arbSampleGeneration;

	protected double affinityThreshold;

	protected CellPool arbMemoryCellPool;
	protected CellPool memoryCellPool;

	// stats
	protected double meanClonesArb;
	protected double meanClonesMemCell;
	protected double meanAllocatedResources;
	protected double meanArbPoolSize;
	protected double meanArbRefinementIterations;
	protected long totalArbDeletions;
	protected long totalMemoryCellReplacements;
	protected long totalArbRefinementIterations;
	protected long totalTrainingInstances;


	public AIRS1Trainer(
			double aAffinityThresholdScalar,
			double aClonalRate,
			double aHyperMutationRate,
			double aMutationRate,
			double aTotalResources,
			double aStimulationValue,
			int aNumInstancesAffinityThreshold,
			Random aRand,
			int aArbCellPoolInitialSize,
			int aMemoryCellPoolInitialSize,
			int aKNN,
			int aDist,
			boolean aUpdate,
			boolean aNegativeSel,
			boolean aUserSpec)
	{
		affinityThresholdScalar = aAffinityThresholdScalar;
		clonalRate = aClonalRate;
		hyperMutationRate = aHyperMutationRate;
		mutationRate = aMutationRate;
		totalResources = aTotalResources;
		stimulationThreshold = aStimulationValue;
		affinityThresholdNumInstances = aNumInstancesAffinityThreshold;
		rand = aRand;
		arbCellPoolInitialSize = aArbCellPoolInitialSize;
		memoryCellPoolInitialSize = aMemoryCellPoolInitialSize;
		kNN = aKNN;

		//added by neda
		Dist = aDist;
		Update = aUpdate;
		NegativeSel = aNegativeSel;
		UserSpec = aUserSpec;
	}


	public void algorithmPreperation(Instances aInstances)
	{
		affinityFunction = new AffinityFunction(aInstances, Dist);
		arbSampleGeneration = prepareSampleGeneration(aInstances);
	}

	protected SampleGenerator prepareSampleGeneration(Instances aInstances)
	{
		return new RandomMutate(rand, aInstances.numClasses(), mutationRate);
	}

	protected void log(String s)
	{
	    System.out.println(s);
	}

	public String getTrainingSummary()
	{
	    StringBuffer buffer = new StringBuffer(1024);
	    NumberFormat f = Utils.format;

	    buffer.append(" - Training Summary - \n");
	    buffer.append("Affinity Threshold:.............................." + f.format(affinityThreshold) + "\n");
	    buffer.append("Total training instances:........................" + f.format(totalTrainingInstances) + "\n");
	    buffer.append("Total memory cell replacements:.................." + f.format(totalMemoryCellReplacements) + "\n");
	    buffer.append("Mean ARB clones per refinement iteration:........" + f.format(meanClonesArb) + "\n");
	    buffer.append("Mean total resources per refinement iteration:..." + f.format(meanAllocatedResources) + "\n");
	    buffer.append("Mean pool size per refinement iteration:........." + f.format(meanArbPoolSize) + "\n");
	    buffer.append("Mean memory cell clones per antigen:............." + f.format(meanClonesMemCell) + "\n");
	    buffer.append("Mean ARB refinement iterations per antigen:......" + f.format(meanArbRefinementIterations) + "\n");
	    buffer.append("Mean ARB prunings per refinement iteration:......" + f.format((double)totalArbDeletions/(double)totalArbRefinementIterations) + "\n");

	    return buffer.toString();
	}


	public AISModelClassifier train(Instances instances)
	throws Exception
	{
		// normalise the dataset
		Normalize normalise = new Normalize();
		normalise.setInputFormat(instances);
		Instances trainingSet = Filter.useFilter(instances, normalise);

                // prepare the algorithm
		algorithmPreperation(trainingSet);


		// calculate affinity threshold
		Utils.calculateAffinityThreshold(trainingSet, affinityThresholdNumInstances, rand, affinityFunction, Dist);

                //read the calculated aff threshold from file
                FileReader f = new FileReader("./AffThreshold/part-r-00000");
                BufferedReader br = new BufferedReader(f);

                String line = br.readLine();
                String[] sarray = line.split(",");
                affinityThreshold = Double.valueOf(sarray[1]);


		// perform the training
		return internalTrain(trainingSet, normalise);
	}


	public void setAffinityThreshold(double a)
	{
		affinityThreshold = a;
	}

	protected AISModelClassifier internalTrain(
					Instances trainingSet,
					Normalize normalise) throws Exception
	{
		// initialise model
		initialise(trainingSet);

                //serialize all objects so we can access them in map functions
                FileOutputStream fos;
		ObjectOutputStream out;
		String filename;

		filename = "normalise.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(normalise);
		out.close();

		filename = "affinityFunction.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(affinityFunction);
		out.close();

		filename = "arbSampleGeneration.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(arbSampleGeneration);
		out.close();

		filename = "memoryCellPool.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(memoryCellPool);
		out.close();

		filename = "arbMemoryCellPool.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(arbMemoryCellPool);
		out.close();


                filename = "dataset.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(trainingSet.instance(0));
		out.close();


		String inputUri  = "hdfs:///FraudDetection/input/";
		String outputUri = "hdfs:///FraudDetection/output/";
		String cacheUri  = "hdfs:///FraudDetection/cache/";

                //delete output folder and empty cache
                Process p = Runtime.getRuntime().exec("hadoop fs -rm " + cacheUri + "/*");
                p.waitFor();

                p = Runtime.getRuntime().exec("hadoop fs -rmr " + outputUri);
                p.waitFor();

                //write normalized files in cache
                p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./normalise.ser " + cacheUri);
                p.waitFor();

                p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./affinityFunction.ser " + cacheUri);
                p.waitFor();

                p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./arbSampleGeneration.ser " + cacheUri);
                p.waitFor();

                p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./memoryCellPool.ser " + cacheUri);
                p.waitFor();

                p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./arbMemoryCellPool.ser " + cacheUri);
                p.waitFor();

                p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./dataset.ser " + cacheUri);
                p.waitFor();



                Configuration conf = new Configuration();
                Job job = new Job(conf);

		job.setJarByClass(AIRS1Trainer.class);
		job.setJobName("Create Detectors");

		FileInputFormat.addInputPath(job, new Path(inputUri));
                FileOutputFormat.setOutputPath(job, new Path(outputUri));

                job.setMapperClass(AIRS1TrainerMapper.class);
                job.setReducerClass(AIRS1TrainerReducer.class);

                job.setOutputKeyClass(Text.class);
                job.setOutputValueClass(IntWritable.class);



                DistributedCache.addCacheFile(new URI(cacheUri+"normalise.ser"), job.getConfiguration());
                DistributedCache.addCacheFile(new URI(cacheUri+"arbSampleGeneration.ser"), job.getConfiguration());
                DistributedCache.addCacheFile(new URI(cacheUri+"memoryCellPool.ser"), job.getConfiguration());
                DistributedCache.addCacheFile(new URI(cacheUri+"arbMemoryCellPool.ser"), job.getConfiguration());
                DistributedCache.addCacheFile(new URI(cacheUri+"affinityFunction.ser"), job.getConfiguration());
                DistributedCache.addCacheFile(new URI(cacheUri+"dataset.ser"), job.getConfiguration());


		//add variables to the configuration so we can use them in map function
		job.getConfiguration().set("affinityThresholdScalar", Double.toString(affinityThresholdScalar));
		job.getConfiguration().set("clonalRate", Double.toString(clonalRate));
		job.getConfiguration().set("hyperMutationRate", Double.toString(hyperMutationRate));
		job.getConfiguration().set("mutationRate", Double.toString(mutationRate));
		job.getConfiguration().set("totalResources", Double.toString(totalResources));
		job.getConfiguration().set("stimulationThreshold", Double.toString(stimulationThreshold));

		job.getConfiguration().setInt("affinityThresholdNumInstances", affinityThresholdNumInstances);
		job.getConfiguration().setInt("arbCellPoolInitialSize", arbCellPoolInitialSize);
		job.getConfiguration().setInt("memoryCellPoolInitialSize", memoryCellPoolInitialSize);
		job.getConfiguration().setInt("kNN", kNN);
		job.getConfiguration().setInt("Dist", Dist);

		job.getConfiguration().setBoolean("Update", Update);
		job.getConfiguration().setBoolean("NegativeSel", NegativeSel);
		job.getConfiguration().setBoolean("UserSpec", UserSpec);

		job.getConfiguration().set("affinityThreshold", Double.toString(affinityThreshold));
		job.getConfiguration().set("meanClonesArb", Double.toString(meanClonesArb));
		job.getConfiguration().set("meanClonesMemCell", Double.toString(meanClonesMemCell));
		job.getConfiguration().set("meanAllocatedResources", Double.toString(meanAllocatedResources));
		job.getConfiguration().set("meanArbPoolSize", Double.toString(meanArbPoolSize));
		job.getConfiguration().set("meanArbRefinementIterations", Double.toString(meanArbRefinementIterations));

		job.getConfiguration().setLong("totalArbDeletions", totalArbDeletions);
		job.getConfiguration().setLong("totalMemoryCellReplacements", totalMemoryCellReplacements);
		job.getConfiguration().setLong("totalArbRefinementIterations", totalArbRefinementIterations);
		job.getConfiguration().setLong("totalTrainingInstances", totalTrainingInstances);


		job.waitForCompletion(true);


                //copy created detectors to the local path
                p = Runtime.getRuntime().exec("hadoop fs -copyToLocal " + outputUri + "/* ./");
                p.waitFor();

		// prepare statistics
		prepareStatistics(trainingSet.numInstances());
		// prepare the classifier
		AISModelClassifier classifier = getClassifier(normalise);

		return classifier;
	}



	protected void prepareStatistics(int aNumTrainingInstances)
	{
		totalTrainingInstances = aNumTrainingInstances;
		meanClonesArb /= totalArbRefinementIterations;
		meanClonesMemCell /= totalTrainingInstances;
		meanAllocatedResources /= totalArbRefinementIterations;
		meanArbPoolSize /= totalArbRefinementIterations;
		meanArbRefinementIterations = ((double)totalArbRefinementIterations / (double)totalTrainingInstances);
	}


        protected AISModelClassifier getClassifier(Normalize aNormalise)
	{
		MajorityVote classifier = new MajorityVote(kNN, aNormalise, memoryCellPool, affinityFunction);
		return classifier;
	}


	protected void initialise(Instances aTrainingSet)
	{
		ModelInitialisation init = getModelInitialisation();
		arbMemoryCellPool = new CellPool(init.generateCellsList(aTrainingSet, arbCellPoolInitialSize));
		memoryCellPool = new CellPool(init.generateCellsList(aTrainingSet, memoryCellPoolInitialSize));
	}


	protected ModelInitialisation getModelInitialisation()
	{
		return new RandomInstancesInitialisation(rand);
	}


	//mapper class
	static class AIRS1TrainerMapper extends Mapper<LongWritable, Text, Text, IntWritable>
        {
		private double affinityThresholdScalar;
		private double clonalRate;
		private double hyperMutationRate;
		private double mutationRate;
		private double totalResources;
		private double stimulationThreshold;
		private int affinityThresholdNumInstances;
		private int arbCellPoolInitialSize;
		private int memoryCellPoolInitialSize;
		private int kNN;
		private int Dist;
		private boolean Update;
		private boolean NegativeSel;
		private boolean UserSpec;


		private AffinityFunction affinityFunction;
		private SampleGenerator arbSampleGeneration;

		private double affinityThreshold;

		private CellPool arbMemoryCellPool;
		private CellPool memoryCellPool;

		// stats
		private double meanClonesArb;
		private double meanClonesMemCell;
		private double meanAllocatedResources;
		private double meanArbPoolSize;
		private double meanArbRefinementIterations;
		private long totalArbDeletions;
		private long totalMemoryCellReplacements;
		private long totalArbRefinementIterations;
		private long totalTrainingInstances;

		private Normalize normalise;

                private Instance dataset;

		public void configure(Configuration job) throws IOException, ClassNotFoundException
		{
			affinityThresholdScalar = Double.valueOf(job.get("affinityThresholdScalar"));
			clonalRate = Double.valueOf(job.get("clonalRate"));
			hyperMutationRate = Double.valueOf(job.get("hyperMutationRate"));
			mutationRate = Double.valueOf(job.get("mutationRate"));
			totalResources = Double.valueOf(job.get("totalResources"));
			stimulationThreshold = Double.valueOf(job.get("stimulationThreshold"));
			affinityThresholdNumInstances = Integer.parseInt(job.get("affinityThresholdNumInstances"));
			arbCellPoolInitialSize = Integer.parseInt(job.get("arbCellPoolInitialSize"));
			memoryCellPoolInitialSize = Integer.parseInt(job.get("memoryCellPoolInitialSize"));
			kNN = Integer.parseInt(job.get("kNN"));

			Dist = Integer.parseInt(job.get("Dist"));
			Update = job.getBoolean("Update", false);
			NegativeSel = job.getBoolean("NegativeSel", false);
			UserSpec = job.getBoolean("UserSpec", false);

			affinityThreshold = Double.valueOf(job.get("affinityThreshold"));

			meanClonesArb = Double.valueOf(job.get("meanClonesArb"));

                        meanClonesMemCell = Double.valueOf(job.get("meanClonesMemCell"));
			meanAllocatedResources = Double.valueOf(job.get("meanAllocatedResources"));
			meanArbPoolSize = Double.valueOf(job.get("meanArbPoolSize"));
			meanArbRefinementIterations = Double.valueOf(job.get("meanArbRefinementIterations"));
			totalArbDeletions = Long.valueOf(job.get("totalArbDeletions"));
			totalMemoryCellReplacements = Long.valueOf(job.get("totalMemoryCellReplacements"));
			totalArbRefinementIterations = Long.valueOf(job.get("totalArbRefinementIterations"));
			totalTrainingInstances = Long.valueOf(job.get("totalTrainingInstances"));


			//read all serialized files from cache
                        Path[] cacheFiles = new Path[0];
                        cacheFiles = DistributedCache.getLocalCacheFiles(job);


                	FileInputStream fis = null;
                	ObjectInputStream in = null;

                        for (Path cacheFile : cacheFiles)
                        {
                            if((cacheFile.toString()).endsWith("normalise.ser"))
                            {
		        	fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	normalise = (Normalize) in.readObject();
		        	in.close();
                            }


                            else if((cacheFile.toString()).endsWith("arbSampleGeneration.ser"))
                            {
		        	fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	arbSampleGeneration = (SampleGenerator) in.readObject();
		        	in.close();

                            }

                            else if((cacheFile.toString()).endsWith("memoryCellPool.ser"))
                            {

		        	fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	memoryCellPool = (CellPool) in.readObject();
		        	in.close();
                            }

                            else if((cacheFile.toString()).endsWith("arbMemoryCellPool.ser"))
                            {

                                fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	arbMemoryCellPool = (CellPool) in.readObject();
		        	in.close();

                            }

                            else if((cacheFile.toString()).endsWith("affinityFunction.ser"))
                            {
     
		        	fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	affinityFunction = (AffinityFunction) in.readObject();
		        	in.close();

                            }

                            else if((cacheFile.toString()).endsWith("dataset.ser"))
                            {
		        	fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	dataset = (Instance) in.readObject();
		        	in.close();

                            }
                        }

		}


                public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
                {

                    if(Integer.parseInt(key.toString()) == 0)
                    {
                    try {
                        configure(context.getConfiguration());}
                    catch (ClassNotFoundException ex) {
                        throw new RuntimeException("Cannot call configure function: "+ex.getMessage(), ex);}

                    }


                    //each map creates some detectors depending to the share it gets
                    String s = Text.decode(value.getBytes());

                    String[] sarray = s.split(",");
                    double[] d = new double[sarray.length];

                    for(int i=0; i<dataset.numAttributes(); i++)
                	//if(!(sarray[i].isEmpty()))
                        {
                            try{
                                d[i] = Double.valueOf(sarray[i]);
                                dataset.setValue(i, d[i]);
                                }
                            catch(Exception e)
                                {}


                        }

                    Instance current = dataset;

                    Cell[] GenCells = createDetectors(current);



                    //convert the cells to string in order to be written in context
                    String[] stCells = new String[2];
                    for(int j=0; j<2; j++)
                    {
                        if(GenCells[j] != null)
                        {
                            stCells[j] = "";
                            for(int i=0; i<GenCells[j].getAttributes().length; i++)
                            {
                                stCells[j] += Double.toString(GenCells[j].getAttributes()[i]);
                                stCells[j] += ",";
                            }

                            stCells[j] += Double.toString(GenCells[j].getAffinity()) + ",";
                            stCells[j] += Double.toString(GenCells[j].getClassification()) + ",";
                            stCells[j] += Double.toString(GenCells[j].getNumResources()) + ",";
                            stCells[j] += Double.toString(GenCells[j].getRate()) + ",";
                            stCells[j] += Double.toString(GenCells[j].getStimulation()) + ",";
                            stCells[j] += Double.toString(GenCells[j].getUser()) + ",";

                            context.write(new Text(stCells[j]), new IntWritable(1));
                        }
                    }

                }//mapper


                 private Cell[] createDetectors(Instance current) throws IOException
                 {
                     int amountIndx = Global.amountIndx;
                     double meanAmount = Global.meanAmount;

                    //defined to return memory cells to be added to model
                    //namely candidate memory cell and best match.
                    Cell[] GenCells = new Cell[2];

                    // train model on current instance
                    // identify best match from memory pool
                    Cell bestMatch = identifyMemoryPoolBestMatch(current);

                    if(bestMatch == null)
                    {
			bestMatch = addNewMemoryCell(current);
                    }

                    // 	never process best match that is identical to the instance
                    else if(bestMatch.getStimulation() == 1.0)
                    {
                    	// do nothing
                    }

                    else
                    {
                        // generate arbs and add to arb pool
			generateARBs(bestMatch, current, amountIndx, meanAmount);

			// get the candidate memory cell
			Cell candidateMemoryCell = runARBRefinement(current);


			// introduce the memory cell
			switch(respondToCandidateMemoryCell(bestMatch, candidateMemoryCell, current))
			{
                            case 0:
                            {
                                GenCells[0] = candidateMemoryCell;
                            }

                            case 1:
                            {
                                GenCells[0] = bestMatch;
				GenCells[1] = candidateMemoryCell;
                            }

                            case 2:
                            {
				GenCells[0] = bestMatch;
                            }
			}
                    }
                    return GenCells;
                 }

                 public Cell identifyMemoryPoolBestMatch(Instance aInstance)
                 {
                    // get memory pool sorted by stimulation
                    LinkedList<Cell> stimulatedSorted = stimulation(memoryCellPool.getCells(),aInstance);
                    // process list until a member of the same class is located

                    for(Cell c: stimulatedSorted)
                    {
                        if(UserSpec)
			{
                            if(aInstance.classValue() == 0)
                            {
                                if(Utils.isSameClass(aInstance, c))
				{
                                    if(aInstance.value(0) == c.getUser())
                                    {
                                        return c;
                                    }
				}
                            }

                            else
                            {
                                if(Utils.isSameClass(aInstance, c))
				{
                                    return c;
				}
                            }
                        }

			else
			{
                            if(Utils.isSameClass(aInstance, c))
                            {
                                return c;
                            }
                        }
                     }


                    return null;
                 }


                 public Cell addNewMemoryCell(Instance aInstance)
                 {
                    // no match, therefore create one
                    Cell c = new Cell(aInstance);
                    // add to memory cell pool
                    memoryCellPool.add(c);
                    return c;
                  }

                 public Cell runARBRefinement(Instance aInstance)
            	 {
			boolean stopCondition = false;
			boolean firstTime = true;
			Cell candidateMemoryCell = null;
			do
			{
        			// perform competition for resources
				candidateMemoryCell = performARBCompetitionForResources(aInstance);
				// calculate if stop condition has been met
				stopCondition = isStoppingCriterion(aInstance);
				// always executed the first time, or when the stop condition is not met
				if(!stopCondition || firstTime)
				{
					LinkedList<Cell> arbs = new LinkedList<Cell>();
					// 3c. variation (mutated clones)
					for(Cell c : arbMemoryCellPool.getCells())
					{
					    arbs.addAll(generateARBVarients(aInstance, c));
					}
					arbMemoryCellPool.add(arbs);
					firstTime = false;
				}

				// stats
				meanArbPoolSize += arbMemoryCellPool.size();
				meanArbRefinementIterations++;
				totalArbRefinementIterations++;
			}
			while(!stopCondition);
			return candidateMemoryCell;
		}

		public void generateARBs(Cell aBestMatchMemoryCell, Instance aInstance, int amountIndx, double meanAmount)
        	{
			// add best match to the arb pool
			arbMemoryCellPool.add(new Cell(aBestMatchMemoryCell));

                        // determine the number of clones to produce
			int numClones = memoryCellNumClones(aBestMatchMemoryCell, aInstance);


			// generate clones
			for (int i = 0; i < numClones; i++)
			{
				// generate mutated clone
				Cell mutatedClone = arbSampleGeneration.generateSample(aBestMatchMemoryCell,aInstance);
				// add to arb pool
				arbMemoryCellPool.add(mutatedClone);
			}

			meanClonesMemCell += numClones;
		}


		public int respondToCandidateMemoryCell(
			Cell bestMatchMemoryCell,
			Cell candidateMemoryCell,
			Instance aInstance) throws IOException
        	{
                  	// recalculate candidate stimulation
                        double candidateStimulation = stimulation(candidateMemoryCell, aInstance);
                        // check if candidate is better
                	if(candidateStimulation > bestMatchMemoryCell.getStimulation())
                        {
                            // add candidate to memory pool
                            memoryCellPool.add(candidateMemoryCell);
                            // check previous best can be removed
                            double affinity = affinityFunction.affinityNormalised(bestMatchMemoryCell, candidateMemoryCell, Dist);
                            if(affinity < getMemoryCellReplacementCutoff())
                            {
    				// remove previous best
    				memoryCellPool.delete(bestMatchMemoryCell);
				totalMemoryCellReplacements++;
				return 0;
                            }
			return 1;
                        }
		return 2;
                }


		protected LinkedList<Cell> stimulation(LinkedList<Cell> cells, Instance aInstance)
		{
		    // calculate stimulation for all the cells
		    for(Cell c : cells)
		    {
		        stimulation(c, aInstance);
		    }
		    // order the population by stimulation
		    Collections.sort(cells, CellPool.stimulationComparator);
		    return cells;
		}

		protected double stimulation(Cell aCell, Instance aInstance)
		{
		    // calculate normalised affinity [0,1]
		    double affinity = affinityFunction.affinityNormalised(aInstance, aCell, Dist);
		    // convert to stimulation
		    double stimulation = 1.0 - affinity;
		    // store
		    aCell.setStimulation(stimulation);
		    // return it in case its needed
		    return stimulation;
		}

		protected Cell performARBCompetitionForResources(Instance aInstance)
		{
			Cell mostStimulatedSameClass = null;
			// calculate stimulation levels
			LinkedList<Cell> sortedStimulated = stimulationNormalisation(arbMemoryCellPool.getCells(),aInstance);
			// normalise stimulation, allocate resources, sum resources for each class
			double [] resources = calculateResourceAllocations(sortedStimulated, aInstance);
			// perform resource management;
        		for (int i = 0; i < resources.length; i++)
			{
				// calculate resources allowed
				double numResAllowed = determineMaximumResourceAllocation(aInstance, i, resources.length);
				// collect all ARBs in this class
				LinkedList<Cell> cells = getAllArbsInClass(i);
				// sort by resource
				Collections.sort(cells, CellPool.resourceComparator);

                                // continue until the resources for this class is below a threshold
				while(resources[i] > numResAllowed)
				{
					double numResourceToRemove = (resources[i]-numResAllowed);

                                        Cell last = cells.getLast();
					// check if element can be removed
					if(last.getNumResources() <= numResourceToRemove)
					{
						cells.removeLast(); // remove from the temp list
						arbMemoryCellPool.delete(last); // remove from the ARB pool
						totalArbDeletions++;
						resources[i] -= last.getNumResources();
					}
					else
					{
						// decrement resources
						double res = last.getNumResources() - numResourceToRemove;
						last.setNumResources(res);
						resources[i] -= numResourceToRemove;
					}
				}

                                // special case of same class as training instance
				if(i == aInstance.classValue())
				{
					// the list is orded by resource allocations, thus the best
					// cell is always at the beginning of the list
					mostStimulatedSameClass = cells.getFirst();
				}
                    }

                    for (int i = 0; i < resources.length; i++)
		    {
                        meanAllocatedResources += resources[i];
		    }

                    return mostStimulatedSameClass;
		}


		protected boolean isStoppingCriterion(Instance aInstance)
		{
				// calculate the mean stimulation level for each class
				int numClasses = aInstance.numClasses();
				double []  meanStimulation = new double[numClasses];
				double [] classCount = new double[numClasses];

				for(Cell c : arbMemoryCellPool.getCells())
				{
					int index = (int) c.getClassification();
					meanStimulation[index] += c.getStimulation();
					classCount[index]++;
				}

				// calculate means - all means must be >= stimulation threshold
				for (int i = 0; i < meanStimulation.length; i++)
				{
					meanStimulation[i] = (meanStimulation[i] / classCount[i]);
					if(meanStimulation[i] < stimulationThreshold)
					{
						return false;
					}
				}

				return true;
			}

	protected double determineMaximumResourceAllocation(
					Instance aInstance,
					int aClassIndex,
					int aNumClasses)
	{
		double numResAllowed = 0.0;

		if(aClassIndex == aInstance.classValue())
		{
			numResAllowed = totalResources / 2.0;
		}
		else
		{
			numResAllowed = totalResources / (2.0 * (aNumClasses-1));
		}

		return numResAllowed;
	}

			protected LinkedList<Cell> getAllArbsInClass(int aClassValue)
			{
				LinkedList<Cell> cells = new LinkedList<Cell>();

				for (Iterator<Cell> iter = arbMemoryCellPool.iterator(); iter.hasNext();)
				{
					Cell c = iter.next();
					if(aClassValue == c.getClassification())
					{
						cells.add(c);
					}
				}

				return cells;
			}


			protected double [] calculateResourceAllocations(
							LinkedList<Cell> list,
							Instance aInstance)
			{
			    double [] resources = new double[aInstance.numClasses()];

				for(Cell c : list)
				{
					// check for not the same class
					if(!Utils.isSameClass(aInstance, c))
					{
						double s = (1.0 - c.getStimulation()); // invert
						c.setStimulation(s);
					}

					double resource = c.getStimulation() * clonalRate;
					c.setNumResources(resource);
					// sum resources

                                        resources[(int)c.getClassification()] += resource;
				}

				return resources;
			}


			protected double getMemoryCellReplacementCutoff()
			{
				return (affinityThreshold * affinityThresholdScalar);
			}

			protected int memoryCellNumClones(Cell aArb, Instance aInstance)
			{
				//alpha is the weight given to each field
				//we calculate it using analysis on the training set
				
				double[][][] alpha = Global.alpha;
				double a = 1;

				if(aInstance.classValue() == 1)
				{
					for(int i=0; i<aInstance.numAttributes()-1; i++)
					{
						for(int k=0; k<alpha[i][0].length; k++)
							if(aInstance.value(i) == alpha[i][0][k])
							{
								a += alpha[i][1][k];
								break;
							}
					}
				}

				return (int) Math.round(aArb.getStimulation() * clonalRate * hyperMutationRate * a);
				
				//return (int) Math.round(aArb.getStimulation() * clonalRate * hyperMutationRate);
			}

			protected LinkedList<Cell> stimulationNormalisation(
			        LinkedList<Cell> cells,
			        Instance aInstance)
			{
				double min = Double.POSITIVE_INFINITY;
				double max = Double.NEGATIVE_INFINITY;

				// determine min and max
				for(Cell c : cells)
				{
					double s = stimulation(c, aInstance);

					if(s < min)
					{
						min = s;
					}
					if(s > max)
					{
						max = s;
					}
				}

				// normalise
				double range = (max - min);
				for(Cell c : cells)
				{
				    double s = c.getStimulation();
				    double normalised = (s-min) / range;
				    c.setStimulation(normalised);

				    // validation
				    if(normalised<0 || normalised>1)
				    {
				        throw new RuntimeException("Normalised stimulation outside range!");
				    }
				}

			    return cells;
			}

			protected LinkedList<Cell> generateARBVarients(Instance aInstance, Cell aArb)
			{
				LinkedList<Cell> newARBs = new LinkedList<Cell>();

				// determine the number of clones to produce
				int numClones = arbNumClones(aArb);
				// generate clones
				for (int i = 0; i < numClones; i++)
				{
					// generate mutated clone
					Cell mutatedClone = arbSampleGeneration.generateSample(aArb, aInstance);

					// add to arb pool
					newARBs.add(mutatedClone);
				}

				meanClonesArb += numClones;

				return newARBs;
			}

			protected int arbNumClones(Cell aArb)
			{
				return (int) Math.round(aArb.getStimulation() * clonalRate);
			}
    }//mapper


	static class AIRS1TrainerReducer extends Reducer<Text, IntWritable, Text, IntWritable>
	{

		public void reduce(Text key, IntWritable value, Context context) throws IOException, InterruptedException
		{
                    //reducer must get the detectors and write them in one file
                    //then the classifier must read them from file to use
                    context.write(key, value);
		}
        }
}
