/*
 * Created on 30/12/2004
 *
 */

/*
 * Edited on 2011/09/19 by Neda Soltani
 * 
 */
package weka.classifiers.immune.airs.algorithm.classification;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.immune.airs.algorithm.AISModelClassifier;
import weka.classifiers.immune.airs.algorithm.AffinityFunction;
import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.CellPool;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.Normalize;
/**
 * Type: MajorityVote
 * File: MajorityVote.java
 * Date: 30/12/2004
 * 
 * Description: 
 * 
 * @author Jason Brownlee
 *
 */
public class MajorityVote extends AISModelClassifier
{
	public MajorityVote(
					int aKNumNeighbours,
					Normalize aNormalise,
					CellPool aCellPool,
					AffinityFunction aAffinityFunction)
	{
		super(aKNumNeighbours, aNormalise, aCellPool, aAffinityFunction);
	}
	
	
	protected int classify(Instance aInstance,int dist, boolean update, boolean user_spec, boolean neg_sel)
	{
            // respond to affinity
            try
            {
                model.readCellsFromFile(aInstance);
            }
            catch (Exception ex)
            {
                Logger.getLogger(MajorityVote.class.getName()).log(Level.SEVERE, null, ex);
            }

            LinkedList<Cell> cells = model.affinityResponseNormalised(aInstance, affinityFunction, dist);

            // determine the majority for the top k cells
            int [] classCounts = determineClassCountForkNN(aInstance, cells, update, user_spec, neg_sel);
		
            int largestIndex = 0;
            int largestCount = classCounts[0];
		
            for (int i = 1; i < classCounts.length; i++)
            {
                if(classCounts[i] > largestCount)
		{
                    largestCount = classCounts[i];
                    largestIndex = i;
                }
            }
            return largestIndex;
	}
}
