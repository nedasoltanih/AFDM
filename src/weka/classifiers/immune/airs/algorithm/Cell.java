/*
 * Created on 30/12/2004
 *
 */
package weka.classifiers.immune.airs.algorithm;

import java.io.Serializable;

import weka.core.Instance;

/**
 * Type: Cell
 * File: Cell.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *  Edited by Neda Soltani on 21/2/2012
 *
 */
public class Cell implements Serializable
{
    private final double [] attributes;
	
    private final int classIndex;
	
	private long usage;
	
	/**
	 * 
	 */
	protected double affinity;
	
	/**
	 * number of resources held by the cell
	 */
	protected double numResources;
	
	/**
	 * current stimulation value
	 */
	protected double stimulation;	
	

	protected double rate;
	
	
	
	
	
	public Cell(double [] aAttributes, int aClassIndex)
	{
		attributes = aAttributes;
		classIndex = aClassIndex;
		
		rate = 0;
	}
	
	public Cell(Instance aInstance)
	{
		// note to double array creates a new object
		this(aInstance.toDoubleArray(), aInstance.classIndex());
		
		rate = 0;
	}
	
	
	public Cell(Cell aCell)
	{
		classIndex = aCell.classIndex;
		attributes = new double[aCell.attributes.length];
		System.arraycopy(aCell.attributes, 0, attributes, 0, attributes.length);
		
		rate = 0;
	}
	
	
	
	public double getClassification()
	{
		return attributes[classIndex];
	}
	
	public double [] getAttributes()
	{
		return attributes;
	}
	
	public int getClassIndex()
	{
		return classIndex;
	}
	
	
	public double getAffinity()
	{
		return affinity;
	}
	public void setAffinity(double affinity)
	{
		this.affinity = affinity;
	}
	
	
	
	public double getNumResources()
	{
		return numResources;
	}
	public void setNumResources(double numResources)
	{
		this.numResources = numResources;
	}
	public double getStimulation()
	{
		return stimulation;
	}
	public void setStimulation(double stimulation)
	{
		this.stimulation = stimulation;
	}
	
	protected long getUsage()
	{
	    return usage;
	}
	protected void incrementUsage()
	{
	    usage++;
	}
	protected void clearUsage()
	{
	    usage = 0;
	}
	
	protected double getUser()
	{
		return this.attributes[0];
	}

        protected void setUser(double user)
	{
		this.attributes[0] = user;
	}
	
	protected void setClassification(double classification)
	{
		attributes[classIndex] = classification;
	}
	
	protected void changeClassification()
	{
		if(attributes[classIndex] == 0)
			attributes[classIndex] = 1;
		else
			attributes[classIndex] = 0;
	}
	
	

	protected double getRate()
	{
		return rate;
	}
	
	protected void decRate(double decSize)
	{
		rate -= decSize;
	}

	protected void incRate(double incSize)
	{
		rate += incSize;
	}
	
	protected void setRate(double s)
	{
		rate = s;
	}

}
