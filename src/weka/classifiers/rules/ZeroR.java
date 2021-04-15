/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.rules;

import global.Global;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Random;
import java.util.Vector;

import sun.security.action.GetBooleanAction;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.immune.airs.AIRSParameterDocumentation;
import weka.classifiers.immune.airs.algorithm.AIRS1Trainer;
import weka.classifiers.immune.airs.algorithm.AISModelClassifier;
import weka.classifiers.immune.airs.algorithm.Cell;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;


/**
 * Type: AIRS1
 * File: AIRS1.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class ZeroR extends Classifier         
	implements OptionHandler, AIRSParameterDocumentation
{
	// paramters	
	protected long seed;	
	protected double affinityThresholdScalar;
	protected double clonalRate;
	protected double hypermutationRate;
	protected double mutationRate;
	protected double totalResources;
	protected double stimulationValue;	
	protected int numInstancesAffinityThreshold;
	protected int arbInitialPoolSize;
	protected int memInitialPoolSize;
	protected int knn;
	
	//added by neda
	protected int distance_funct;	//which distance funtion to use
	protected boolean update; // to use update while test or not
	protected boolean negative_sel;	//to use negative selection in training or not
	protected boolean user_spec;	//to have user specific cells or not
	
	protected String trainingSummary;
	protected String classifierSummary;
	
	//added by neda
	//protected AIRS1Trainer trainer;
	
	
	private final static String [] PARAMETERS =
    {
        "S", // seed
		"F", // affinity threshold
		"C", // clonal rate
		"H", // hypermutation
		"M", // mutation rate
		"R", // total resources
		"V", // stimulation value
		"A", // num affinity threshold instances
		"B", // arb pool size
		"E", // mem pool size
		"K",  // kNN
		
		//added by neda
		"DIST", //distance_funct
		"UPDATE", 	//update
		"NEG",		//negative_sel
		"USER"	//user_spec
    };
	
	
	private final static String [] DESCRIPTIONS =
    {
		PARAM_SEED,
		PARAM_ATS,
		PARAM_CLONAL_RATE,
		PARAM_HMR,
		PARAM_MUTATION_RATE,
		PARAM_RESOURCES,
		PARAM_STIMULATION,
		PARAM_AT_INSTANCES,
		PARAM_ARB_INSTANCES,
		PARAM_MEM_INSTANCES,
		PARAM_KNN,
		
		//added by neda
		PARAM_DIST,
		PARAM_UPDATE,
		PARAM_NEGSEL,
		PARAM_USER
    };
	
	
	/**
	 * The model
	 */
	protected AISModelClassifier classifier;
	
	

	public ZeroR()
	{
			/*
		
		// set default values
		seed = 1;		
		affinityThresholdScalar = 0.0;
		mutationRate = 0.1;
		totalResources = 177.0;
		stimulationValue = 1.0;		
		clonalRate = 30;
		hypermutationRate = 10;		
		numInstancesAffinityThreshold = -1;
		arbInitialPoolSize = 1;
		memInitialPoolSize = 5;
		knn = 1;
		*/
		
		
		// set default values
		seed = 1;		
		affinityThresholdScalar = 0.2;
		mutationRate = 0.1;
		totalResources = 150;
		stimulationValue = 0.9;		
		clonalRate = 10;
		hypermutationRate = 2.0;		
		numInstancesAffinityThreshold = -1;
		arbInitialPoolSize = 1;
		memInitialPoolSize = 5;
		knn = 3;
		
		//added by neda
		distance_funct=1;
		update = false;
		negative_sel = false;
		user_spec = false;
		
		
	}
	
	

	/**
	 * @param data
	 * @throws java.lang.Exception
	 */
	public void buildClassifier(Instances data) throws Exception
	{
		Instances trainingInstances = new Instances(data);

		// must have a class assigned
		if (trainingInstances.classIndex() < 0)
		{
			throw new Exception("No class attribute assigned to instances");
		}
		// class must be nominal
		else if(!trainingInstances.classAttribute().isNominal())
		{
			throw new UnsupportedClassTypeException("Class attribute must be nominal");
		}
		// must have attributes besides the class attribute
		else if(trainingInstances.numAttributes() <= +1)
		{
			throw new Exception("Dataset contains no supported comparable attributes");
		}
		
		// delete with missing class
		trainingInstances.deleteWithMissingClass();
		for (int i = 0; i < trainingInstances.numAttributes(); i++)
		{
			trainingInstances.deleteWithMissing(i);
		}
		
		// must have some training instances
		if (trainingInstances.numInstances() == 0)
		{
			throw new Exception("No usable training instances!");
		}
		
		// validate paramters
		validateParameters(trainingInstances);

		// construct trainer
		Random rand = new Random(seed);
                AIRS1Trainer trainer = new AIRS1Trainer(
						affinityThresholdScalar,
						clonalRate,
						hypermutationRate,
						mutationRate,
						totalResources,
						stimulationValue,
						numInstancesAffinityThreshold,
						rand, 
						arbInitialPoolSize,
						memInitialPoolSize,
						knn,

						//added by neda
						distance_funct,
						update,
						negative_sel,
						user_spec);		
		// prepare classifier
		/*
		//{added by neda
		//write the training set in file
		BufferedWriter bw = new BufferedWriter(new FileWriter("training set.txt",true));	
		
		for(int i=0; i<trainingInstances.numInstances(); i++)
		{
			for(int j=0; j<trainingInstances.instance(i).numAttributes(); j++)
			{
				bw.write(Double.toString(trainingInstances.instance(i).value(j)) + ",");
			}			
			bw.write("\r\n");
		}
		
                bw.close();
		//added by neda}
		*/

                
		
		classifier = trainer.train(trainingInstances);
		
		// get summaries
		trainingSummary = trainer.getTrainingSummary();
		classifierSummary = classifier.getModelSummary(trainingInstances);
	}
	
	
	protected void validateParameters(Instances trainingInstances)
		throws Exception
	{		
		int numInstances = trainingInstances.numInstances();		

		if(arbInitialPoolSize > numInstances)
		{
		    arbInitialPoolSize = numInstances;
		}
		else if(memInitialPoolSize > numInstances)
		{
		    memInitialPoolSize = numInstances;
		}
	}
	

	public double classifyInstance(Instance instance) 
		throws Exception 
	{
		if(classifier == null)
		{
			throw new Exception("Algorithm has not been prepared.");
		}
		
		// TODO: validate of data provided matches training data specs
		
		double classified = classifier.classifyInstance(instance, distance_funct, update, user_spec, negative_sel);
		
		if(update && instance != null)
		{
			// normalise vector
			try {
				classifier.getNormalizer().input(instance);
			} catch (Exception e) {
				throw new RuntimeException("Unable to classify instance: "+e.getMessage(), e);
			}
			instance = classifier.getNormalizer().output();
			
			//updateClassifier(instance);
		}
		
		return classified;
	}
	
	
	
	public String toString()
	{
		StringBuffer buffer = new StringBuffer(1000);		
		buffer.append("AIRS - Artificial Immune Recognition System v1.0.\n");		
		buffer.append("\n");
		 
		if(trainingSummary != null)
		{		    
		    buffer.append(trainingSummary);
		    buffer.append("\n");
		}
		
		if(classifierSummary != null)
		{		 
		    buffer.append(classifierSummary);
		}
		
		return buffer.toString();
	}
	
	
    public String globalInfo()
    {
		StringBuffer buffer = new StringBuffer(1000);
		buffer.append(toString());
		buffer.append("A resource limited artifical immune system (AIS) ");
		buffer.append("for supervised classification, using clonal selection, ");
		buffer.append("affinity maturation and affinity recognition balls (ARBs).");
		buffer.append("\n\n");

		buffer.append("Andrew B. Watkins, ");
		buffer.append("A resource limited artificial immune classifier, ");
		buffer.append("Mississippi State University, ");
		buffer.append("(Masters Thesis), 2001.");
		
		return buffer.toString();
    }
	
	public Enumeration listOptions()
	{
		Vector<Option> list = new Vector<Option>(15);
		
		// add parents options
		Enumeration e = super.listOptions();
		while(e.hasMoreElements())
		{
			list.add((Option)e.nextElement());
		}
		
		// add new options
		for (int i = 0; i < PARAMETERS.length; i++)
		{
			Option o = new Option(DESCRIPTIONS[i], PARAMETERS[i], 1, "-"+PARAMETERS[i]);
			list.add(o);
		}
		
		return list.elements();
	}
	
	
	public void setOptions(String[] options) throws Exception
	{
		// parental option setting
		super.setOptions(options);
	    // long
	    setSeed(getLong(PARAMETERS[0], options));
	    // double
	    setAffinityThresholdScalar(getDouble(PARAMETERS[1], options));
	    setClonalRate(getDouble(PARAMETERS[2], options));
	    setHypermutationRate(getDouble(PARAMETERS[3], options));
	    setMutationRate(getDouble(PARAMETERS[4], options));
	    setTotalResources(getDouble(PARAMETERS[5], options));
	    setStimulationValue(getDouble(PARAMETERS[6], options));
	    // int
	    setNumInstancesAffinityThreshold(getInteger(PARAMETERS[7], options));
	    setArbInitialPoolSize(getInteger(PARAMETERS[8], options));
	    setMemInitialPoolSize(getInteger(PARAMETERS[9], options));
	    setKnn(getInteger(PARAMETERS[10], options));
	    
	  //added by neda
	    setDist(getInteger(PARAMETERS[11], options));
	    setUp(getBoolean(PARAMETERS[12], options));
	    setNeg(getBoolean(PARAMETERS[13], options));
	    setUser(getBoolean(PARAMETERS[14], options));
	}
	
	
	//{added by neda
	private boolean getBoolean(String param, String[] options) throws Exception 
	{
		String value = Utils.getOption(param.charAt(0), options);
		if(value == null)
		{
			throw new Exception("Parameter not provided: " + param);
		}
		
		return Boolean.parseBoolean(value);
	}
	//added by neda}


	protected double getDouble(String param, String [] options)
		throws Exception
	{
		String value = Utils.getOption(param.charAt(0), options);
		if(value == null)
		{
			throw new Exception("Parameter not provided: " + param);
		}
		
		return Double.parseDouble(value);
	}
	protected int getInteger(String param, String [] options)
		throws Exception
	{
		String value = Utils.getOption(param.charAt(0), options);
		if(value == null)
		{
			throw new Exception("Parameter not provided: " + param);
		}
		
		return Integer.parseInt(value);
	}
	protected long getLong(String param, String [] options)
		throws Exception
	{
		String value = Utils.getOption(param.charAt(0), options);
		if(value == null)
		{
			throw new Exception("Parameter not provided: " + param);
		}
		
		return Long.parseLong(value);
	}
	
	

    public String[] getOptions()
    {
    	LinkedList<String> list = new LinkedList<String>();
    	
        String [] options = super.getOptions();
        for (int i = 0; i < options.length; i++)
		{
			list.add(options[i]);
		}
        
        // long
        list.add("-"+PARAMETERS[0]);
        list.add(Long.toString(seed));
        // double
        list.add("-"+PARAMETERS[1]);
        list.add(Double.toString(affinityThresholdScalar));
        list.add("-"+PARAMETERS[2]);
        list.add(Double.toString(clonalRate));
        list.add("-"+PARAMETERS[3]);
        list.add(Double.toString(hypermutationRate));
        list.add("-"+PARAMETERS[4]);
        list.add(Double.toString(mutationRate));
        list.add("-"+PARAMETERS[5]);
        list.add(Double.toString(totalResources));
        list.add("-"+PARAMETERS[6]);
        list.add(Double.toString(stimulationValue));
        // int
        list.add("-"+PARAMETERS[7]);
        list.add(Integer.toString(numInstancesAffinityThreshold));
        list.add("-"+PARAMETERS[8]);
        list.add(Integer.toString(arbInitialPoolSize));
        list.add("-"+PARAMETERS[9]);
        list.add(Integer.toString(memInitialPoolSize));
        list.add("-"+PARAMETERS[10]);
        list.add(Integer.toString(knn));
        
        //added by neda
        list.add("-"+PARAMETERS[11]);
        list.add(Integer.toString(distance_funct));
        list.add("-"+PARAMETERS[12]);
        list.add(Boolean.toString(update));
        list.add("-"+PARAMETERS[13]);
        list.add(Boolean.toString(negative_sel));
        list.add("-"+PARAMETERS[14]);
        list.add(Boolean.toString(user_spec));
        
        
        return list.toArray(new String[list.size()]);
    }
    
    
	// long
	public String seedTipText(){return DESCRIPTIONS[0];}
	// double
	public String affinityThresholdScalarTipText(){return DESCRIPTIONS[1];}
	public String clonalRateTipText(){return DESCRIPTIONS[2];}
	public String hypermutationRateTipText(){return DESCRIPTIONS[3];}
	public String mutationRateTipText(){return DESCRIPTIONS[4];}
	public String totalResourcesTipText(){return DESCRIPTIONS[5];}
	public String stimulationValueTipText(){return DESCRIPTIONS[6];}
	// int
	public String numInstancesAffinityThresholdTipText(){return DESCRIPTIONS[7];}
	public String arbInitialPoolSizeTipText(){return DESCRIPTIONS[8];}
	public String memInitialPoolSizeTipText(){return DESCRIPTIONS[9];}
	public String knnTipText(){return DESCRIPTIONS[10];}
	
	//added by neda
	public String distance_functTipText(){return DESCRIPTIONS[11];}
	public String updateTipText(){return DESCRIPTIONS[12];}
	public String negative_selTipText(){return DESCRIPTIONS[13];}
	public String user_specTipText(){return DESCRIPTIONS[14];}
    
    
    
	public double getAffinityThresholdScalar()
	{
		return affinityThresholdScalar;
	}
	public void setAffinityThresholdScalar(double affinityThresholdScalar)
	{
		this.affinityThresholdScalar = affinityThresholdScalar;
	}
	public int getArbInitialPoolSize()
	{
		return arbInitialPoolSize;
	}
	public void setArbInitialPoolSize(int arbInitialPoolSize)
	{
		this.arbInitialPoolSize = arbInitialPoolSize;
	}
	public AISModelClassifier getClassifier()
	{
		return classifier;
	}
	public void setClassifier(AISModelClassifier classifier)
	{
		this.classifier = classifier;
	}
	public double getClonalRate()
	{
		return clonalRate;
	}
	public void setClonalRate(double clonalRate)
	{
		this.clonalRate = clonalRate;
	}
	public double getHypermutationRate()
	{
		return hypermutationRate;
	}
	public void setHypermutationRate(double hypermutationRate)
	{
		this.hypermutationRate = hypermutationRate;
	}
	public int getKnn()
	{
		return knn;
	}
	public void setKnn(int knn)
	{
		this.knn = knn;
	}
	
	//{added by neda
	public int getDist()
	{
		return distance_funct;
	}
	public void setDist(int distance_funct)
	{
		this.distance_funct = distance_funct;
	}
	public boolean getUp()
	{
		return update;
	}
	public void setUp(boolean update)
	{
		this.update = update;
	}
	public boolean getNeg()
	{
		return negative_sel;
	}
	public void setNeg(boolean negative_sel)
	{
		this.negative_sel = negative_sel;
	}
	public boolean getUser()
	{
		return user_spec;
	}
	public void setUser(boolean user_spec)
	{
		this.user_spec = user_spec;
	}
	//added by neda}
	
	
	public int getMemInitialPoolSize()
	{
		return memInitialPoolSize;
	}
	public void setMemInitialPoolSize(int memInitialPoolSize)
	{
		this.memInitialPoolSize = memInitialPoolSize;
	}
	public double getMutationRate()
	{
		return mutationRate;
	}
	public void setMutationRate(double mutationRate)
	{
		this.mutationRate = mutationRate;
	}
	public int getNumInstancesAffinityThreshold()
	{
		return numInstancesAffinityThreshold;
	}
	public void setNumInstancesAffinityThreshold(int numInstancesAffinityThreshold)
	{
		this.numInstancesAffinityThreshold = numInstancesAffinityThreshold;
	}
	public long getSeed()
	{
		return seed;
	}
	public void setSeed(long seed)
	{
		this.seed = seed;
	}
	public double getStimulationValue()
	{
		return stimulationValue;
	}
	public void setStimulationValue(double stimulationValue)
	{
		this.stimulationValue = stimulationValue;
	}
	public double getTotalResources()
	{
		return totalResources;
	}
	public void setTotalResources(double totalResources)
	{
		this.totalResources = totalResources;
	}
	
	
	public static void main(String[] argv)
	{

		try
		{
			System.out.println(Evaluation.evaluateModel(new ZeroR(), argv));
		}
		catch (Exception e)
		{
			System.err.println(e.getMessage());
		}
	}
	
	/*
	//added by neda
	public void updateClassifier(Instance aInstance) throws IOException
	{
		int amountIndx = Global.amountIndx;
		double meanAmount = Global.meanAmount;

		
		Cell bestMatch = trainer.identifyMemoryPoolBestMatch(aInstance);
		if(bestMatch == null)
		{
		    bestMatch = trainer.addNewMemoryCell(aInstance);
		}
		// never process best match that is identical to the instance
		else if(bestMatch.getStimulation() == 1.0)
		{
			// do nothing
		}
		else
		{
			// generate arbs and add to arb pool
			trainer.generateARBs(bestMatch, aInstance, amountIndx, meanAmount);
			// get the candidate memory cell
			Cell candidateMemoryCell = trainer.runARBRefinement(aInstance);
			// introduce the memory cell
			trainer.respondToCandidateMemoryCell(bestMatch, candidateMemoryCell, aInstance);
		}
		
		
	}*/
}
