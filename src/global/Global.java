package global;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;



public class Global {

	public static double AIRSthreshold = 1;
	public static double cspra_dist = 0.0;
	public static double airs_dist = 0.0;
	public static double[] w = new double [18];
	
	//the percentage of fraud for each value of each field
	public static double[][][] alpha;
	
	public static double meanAmount = 0.0;
	public static int amountIndx = 4;
	
	
	
	
	public static void setAlpha(Instances instances)
	{
		Instances data = new Instances(instances);
		
		alpha = new double[instances.numAttributes()][2][];
		// the number of values for each field
		int[] numValues = new int[instances.numAttributes()];
		
		for(int i=0; i<data.numAttributes(); i++)
		{
			data.sort(i);
			numValues[i] = 1;

			for(int j=0; j<data.numInstances()-1; j++)
			{
				if(data.instance(j).value(i) != data.instance(j+1).value(i))
				{
					numValues[i]++;
				}
			}
			
			alpha[i][0] = new double[numValues[i]];
			alpha[i][1] = new double[numValues[i]];
			
			int k=0;
			
			
			for(int j=0; j<data.numInstances(); j++)
			{
				if(data.instance(j).classValue() == 1.0)
					alpha[i][1][k]++;

				if(k < numValues[i]-1)
					if(data.instance(j).value(i) != data.instance(j+1).value(i))
					{
						alpha[i][0][k] = data.instance(j).value(i);	
						k++;
					}
				if(j==0 || k==numValues[i]-1)
					alpha[i][0][k] = data.instance(j).value(i);

			}
			
		}
		
		for(int i=0; i<data.numAttributes()-1; i++)
			for(int j=0; j<numValues[i]; j++)
				alpha[i][1][j] /= alpha[data.classIndex()][1][1];
	}

}

