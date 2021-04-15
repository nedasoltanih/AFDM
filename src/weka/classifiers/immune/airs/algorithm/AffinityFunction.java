/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Type: AffinityFunction
 * File: AffinityFunction.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 * Edited by Neda Soltani on 21/2/2012
 */

//{completely edited by neda by adding dist

public class AffinityFunction extends DistanceFunction
implements Serializable
{		
	public AffinityFunction(Instances aInstances, int dist)
	{
		super(aInstances, dist);
	}
	
	public double affinityNormalised(double [] i1, double [] i2, int dist)
	{
		// single point for adjustment
		return distanceEuclideanNormalised(i1, i2, dist);
	}	
	public double affinityUnnormalised(double [] i1, double [] i2, int dist)
	{
		// single point for adjustment
		return distanceEuclideanUnnormalised(i1, i2, dist);
	}
	
	
	
	
	public double affinityNormalised(Instance i1, Instance i2, int dist)
	{
		return affinityNormalised(i1.toDoubleArray(), i2.toDoubleArray(), dist);
	}	
	public double affinityNormalised(Instance i1, Cell c2,int dist)
	{
		return affinityNormalised(i1.toDoubleArray(), c2.getAttributes(), dist);
	}	
	public double affinityNormalised(double [] i1, Cell c2,int dist)
	{
		return affinityNormalised(i1, c2.getAttributes(), dist);
	}
	public double affinityNormalised(Cell c1, Cell c2,int dist)
	{
		return affinityNormalised(c1.getAttributes(), c2.getAttributes(), dist);
	}
	
	
	public double affinityUnnormalised(Instance i1, Instance i2,int dist)
	{
		return affinityUnnormalised(i1.toDoubleArray(), i2.toDoubleArray(), dist);
	}	
	public double affinityUnnormalised(Instance i1, Cell c2,int dist)
	{
		return affinityUnnormalised(i1.toDoubleArray(), c2.getAttributes(), dist);
	}	
	public double affinityUnnormalised(double [] i1, Cell c2,int dist)
	{
		return affinityUnnormalised(i1, c2.getAttributes(), dist);
	}
	public double affinityUnnormalised(Cell c1, Cell c2,int dist)
	{
		return affinityUnnormalised(c1.getAttributes(), c2.getAttributes(), dist);
	}


        public double affinityNormalised2(Instance i1, Instance i2, int dist, double MaxDist)
	{
		return distanceEuclideanNormalised2(i1.toDoubleArray(), i2.toDoubleArray(), dist, MaxDist);
	}
	
}
