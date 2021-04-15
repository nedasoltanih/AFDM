/*
 * Created on 8/01/2005
 */
package weka.classifiers.immune.airs.algorithm.merge;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.immune.airs.algorithm.AISModelClassifier;
import weka.classifiers.immune.airs.algorithm.AffinityFunction;
import weka.classifiers.immune.airs.algorithm.Cell;
import weka.classifiers.immune.airs.algorithm.CellPool;
import weka.classifiers.immune.airs.algorithm.MemoryCellMerger;
import weka.classifiers.immune.airs.algorithm.Utils;
import weka.classifiers.immune.airs.algorithm.classification.MajorityVote;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * Type: ConcatonateMerge <br>
 * File: ConcatonateMerge.java <br>
 * Date: 8/01/2005 <br>
 * <br>
 * Description: <br>
 * 
 * @author Jason Brownlee
 */
public class PruneMerge implements MemoryCellMerger
{

	/**
	 * @param cells
	 * @return
	 */
	public AISModelClassifier mergeMemoryCells(
					LinkedList [] cells,
					int aKNN, 
					Normalize aNormalise, 
					AffinityFunction aFunction,
					Instances aDataset)
	{
		LinkedList<Cell> masterList = new LinkedList<Cell>();
		
		for (int i = 0; i < cells.length; i++)
		{
			masterList.addAll(cells[i]);
		}
		
		CellPool pool = new CellPool(masterList);		
        try {
            // perform classification and pruning with dataset
            Utils.performPrunning(pool, aDataset, aFunction, 1);
        } catch (Exception ex) {
            Logger.getLogger(PruneMerge.class.getName()).log(Level.SEVERE, null, ex);
        } 
		
		MajorityVote classifier = new MajorityVote(aKNN, aNormalise, pool, aFunction);
		return classifier;
	}


}