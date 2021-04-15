
package weka.classifiers.immune.airs;

import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
/**
 * Type: AIRS1Test
 * Date: 6/01/2005
 * 
 * 
 * @author Jason Brownlee
 */
public class AIRS1Test extends AIRSAlgorithmTester
{

	public static void main(String[] args)
	{
		try
		{
			AIRSAlgorithmTester tester = new AIRS1Test();
			tester.run();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	protected void setSeed(Classifier aClassifier, long aSeed)
	{
		((ZeroR)aClassifier).setSeed(aSeed);
	}
	
	protected Classifier getAIRSAlgorithm()
	{
		ZeroR algorithm = new ZeroR();
		return algorithm;
	}
}
