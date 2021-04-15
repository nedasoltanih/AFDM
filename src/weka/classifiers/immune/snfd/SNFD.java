package weka.classifiers.immune.snfd;

import java.util.Random;

import global.Global;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.immune.airs.AIRS1;
import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.cspra.CSPRA2;
import weka.classifiers.immune.cspra.CSPRA;
import weka.classifiers.immune.airs.algorithm.AIRS1Trainer;

public class SNFD extends Classifier 
{	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;



	//CSPRA classifier 
	protected CSPRA cspra_;
	
	//AIRS Classifier
	protected AIRS1 airs_;
	
	protected int version;
	
	public void buildClassifier(Instances data) throws Exception 
	{
		version = 1;
		
		cspra_ = new CSPRA();
		airs_ = new AIRS1();
		
		cspra_.buildClassifier(data);
		airs_.buildClassifier(data);
	}
	
	public double classifyInstance(Instance instance) 
	throws Exception 
	{	
		
		double finalClassValue = 0.0, cspra_predClass, airs_predClass;					

		///////////		
		if(version == 1)
		{
			airs_predClass = airs_.classifyInstance(instance);		
		
			if(airs_predClass == 1.0) //RECORD IS SELF
				finalClassValue = airs_predClass;
		
			else
			{
				cspra_predClass = cspra_.classifyInstance(instance);
				finalClassValue = cspra_predClass;
			}
		}
	
		///////////		
		else if(version == 2)
		{

			cspra_predClass = cspra_.classifyInstance(instance);
		
			if(cspra_predClass == 0.0) //RECORD IS SELF
				finalClassValue = cspra_predClass;
		
			else
			{
				airs_predClass = airs_.classifyInstance(instance);
				finalClassValue = airs_predClass;
			}
		}
		
		
		else // version == 3
		{
			cspra_predClass = cspra_.classifyInstance(instance);
			airs_predClass = airs_.classifyInstance(instance);
			
			if(airs_predClass != cspra_predClass)
			{
				if(Global.airs_dist < Global.cspra_dist)
					finalClassValue = airs_predClass;
				
				else if(Global.airs_dist > Global.cspra_dist)
					finalClassValue = cspra_predClass;
				
				else //if the distances are equal randomly chose one
				{
					Random r = new Random();
					double d = r.nextDouble();
					
					if(d > 0.5)
						finalClassValue = cspra_predClass;
					else
						finalClassValue = airs_predClass;
				}
			}
			
			else
				finalClassValue = airs_predClass;
		}
		return finalClassValue;
	}

	
	public String toString()
	{
		StringBuffer buffer = new StringBuffer(1000);		
		buffer.append("SNFD" + Double.toString(version));		
		buffer.append("\n");
		 
		
		return buffer.toString();
	}
	
}
