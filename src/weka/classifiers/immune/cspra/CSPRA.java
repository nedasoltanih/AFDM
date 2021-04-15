/*Date: 2012/2/21
 * Author Neda Soltani
 * */


package weka.classifiers.immune.cspra;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;


public class CSPRA extends Classifier 
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected Normalize normaliser;
	
	//threshold for APC detector
	protected static	double APCThreshold;	
	
	//number of APC detectors
	protected static	int No_APC;
	
	//APC Detectors
	protected static	double[][] APC_det;
	
	//Number of Samples
	protected static 	int No_Samples;
	
	//Number of fields in dataset
	protected static 	int No_Fields;	
	
	protected static double maxDistance;
	
	
	//the classifier
	protected CSPRAClassifier classifier; 
	
	
	
	
	public CSPRA()
	{		
		maxDistance =  Double.NEGATIVE_INFINITY;
		APCThreshold = 0;
	}

	public void CSPRAInitialization(Instances instances) throws Exception
	{
		// normalise the dataset
		normaliser = new Normalize();
		normaliser.setInputFormat(instances);
		Instances trainingSet = Filter.useFilter(instances, normaliser);
			
		No_Fields = instances.numAttributes();
		No_Samples = instances.numInstances();		

		GenerateAPCDetector(trainingSet);   //{{loc1, min, max, mean}, {loc2, min, max, mean}, ...}	
	}

	
	protected void GenerateAPCDetector(Instances instances) throws IOException 
	{
		APC_det = new double[No_Fields-1][4];
		
		int tt=0;
		
		double min, max, mean;
		int cnt=0;
		
		for(int ii=0; ii<No_Fields-1; ii++ )
		{			
			//compute min, max and mean for selected indexes
			min = Double.POSITIVE_INFINITY;
			max = Double.NEGATIVE_INFINITY;
			mean = 0;
				
			for (int jj=0; jj<No_Samples; jj++)
			{
				if(instances.instance(jj).classValue() == 0.0)
				{						
					if(instances.instance(jj).value(ii) < min)
						min = instances.instance(jj).value(ii);
			
					if(instances.instance(jj).value(ii) > max)
						max = instances.instance(jj).value(ii);
			
					mean += instances.instance(jj).value(ii);
					cnt++;
				}
			}
			
			mean /= cnt;
			
			APC_det[tt][0] = ii;
			APC_det[tt][1] = min;
			APC_det[tt][2] = max;
			APC_det[tt][3] = mean;
    	
			tt++;
		}					  
			
		No_APC = tt;	
		
		ComputeMaxDistance(instances);
		APCThreshold = maxDistance * 0.8;
		
		
		classifier = new CSPRAClassifier(APC_det, APCThreshold, No_APC);
		
		
		//write all detectors in a file
		BufferedWriter bw = new BufferedWriter(new FileWriter("APC Detectors.txt",true));			
		
		double det;
		for(int j=0; j<No_APC; j++)// Cell det : memoryCellPool.cells)
		{			  
			for(int i=0; i<4; i++)
			{
				det = (double)((int)(APC_det[j][i] * 10000)) / 10000;
				bw.write(Double.toString(det) + ",");    			
			}    	
			
			bw.write("\r\n");
		}
		
		bw.close();
		
		
	}


	private void ComputeMaxDistance(Instances instances) 
	{
		maxDistance = Double.NEGATIVE_INFINITY;
		double dist;
		int jj, ii, tt;		
		
		for(tt=0; tt<No_Samples; tt++)
		{
			dist = 0.0;
			
			if(instances.instance(tt).classValue() == 0.0)
			{			
				for(ii=0; ii<No_APC; ii++)
				{				
					jj = (int) (APC_det[ii][0]);
					
					if(APC_det[ii][2]-APC_det[jj][1] != 0)
						dist += (Math.abs(instances.instance(tt).value(jj)-APC_det[ii][3]))/(APC_det[ii][2]-APC_det[jj][1]);
					else
						dist += (Math.abs(instances.instance(tt).value(jj)-APC_det[ii][3]));
				}
					    
				if(dist > maxDistance)
					maxDistance = dist;
			}	
		}		
	
	}


	private double CheckWithAPCDetector(Instance currentInstance) 
	{	
		double dist = 0;
		int jj = 0, ii = 0;
		
		//compute the distance with equation (1) in section 5.2
		for(ii=0; ii<No_APC; ii++)
		{				
			jj = (int) (APC_det[ii][0]);
			if(APC_det[ii][2]-APC_det[jj][1] != 0)
				dist += (Math.abs(currentInstance.value(jj)-APC_det[ii][3]))/(APC_det[ii][2]-APC_det[jj][1]);
			else
				dist += Math.abs(currentInstance.value(jj)-APC_det[ii][3]);
		}	    		
		
	    if(dist < APCThreshold)	    
	    	return 0.0; 	       
			
		else
			return 1.0;
	}


	@Override
	public void buildClassifier(Instances data) throws Exception 
	{
		CSPRAInitialization(data);
	}
	
	public double classifyInstance(Instance instance) 
	throws Exception 
	{
		if(classifier == null)
		{
			throw new Exception("Algorithm has not been prepared.");
		}
	
		try
		{
			normaliser.input(instance);
		}
		
		catch (Exception e) 
		{
			throw new RuntimeException("Unable to classify instance: "+e.getMessage(), e);
		}
		
		instance = normaliser.output();
		
		return CheckWithAPCDetector(instance);
	}

	public String toString()
	{
		StringBuffer buffer = new StringBuffer(1000);		
		buffer.append("CSPRA");		
		buffer.append("\n");
		 
		
		return buffer.toString();
	}
	
}