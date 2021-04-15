/*Date: 2012/2/21
 * Author Neda Soltani
 * */


package weka.classifiers.immune.cspra;

import java.io.Serializable;
import weka.core.Instances;


public class CSPRAClassifier implements Serializable
{
	protected final double[][] APC_det;
	protected final double APCTreshold;
	protected final int APC_No;
	
	public CSPRAClassifier(
			double[][] detectors,
			double threshold,
			int APCCount)
	{
		APC_det = detectors;
		APCTreshold = threshold;
		APC_No = APCCount;
	}
	
	public double[][] getAPC_det()
	{
		return APC_det;
	}
	
	public double getAPCThreshold()
	{
		return APCTreshold;
	}
	
	public String getModelSummary(Instances aInstances)
	{
	    StringBuffer buffer = new StringBuffer(1024);
	    
	    buffer.append(" - Classifier APC Detectors - \n");
	    buffer.append(this.APC_No + "\n");
	    
	    return buffer.toString();
	}
}
