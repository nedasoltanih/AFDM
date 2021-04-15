/*
 * Created on 30/12/2004
 * Last Edited on 21/2/2012
 * map/reduce version
 *
 */
package weka.classifiers.immune.airs.algorithm;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URISyntaxException;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

import java.net.URI;
import java.util.ArrayList;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
/**
 * Type: Utils
 * File: Utils.java
 * Date: 30/12/2004
 * Last Edit: 21/2/2012
 *
 * Description:
 *
 * @author Jason Brownlee
 * @Edited Neda Soltani
 *
 */
public final class Utils
{
    public final static NumberFormat format = new DecimalFormat();



	public final static boolean isSameClass(Instance aInstance, Cell aCell)
	{
		return aInstance.classValue() == aCell.getClassification();
	}



	public final static void calculateAffinityThreshold(
					Instances aInstances,
					int affinityThresholdNumInstances,
					Random rand,
					AffinityFunction affinityFunction,
					int dist) throws IOException, InterruptedException, URISyntaxException, ClassNotFoundException
	{
		Instances newset = new Instances(aInstances);

		// check if all should be used
		if(affinityThresholdNumInstances < 1 || affinityThresholdNumInstances > newset.numInstances())
		{
		    affinityThresholdNumInstances = newset.numInstances();
		}
		// prune some
		else if(newset.numInstances() > affinityThresholdNumInstances)
		{
			// randomise the dataset
			newset.randomize(rand);

			while(newset.numInstances() > affinityThresholdNumInstances)
			{
				newset.delete(0);
			}
		}

                //split and write training set on file
                BufferedWriter bw = null;
                int t=0, splitSize = 3000;
                for(t=0; t<Math.ceil((double)(newset.numInstances())/splitSize); t++)
                {
                    bw = new BufferedWriter(new FileWriter("./trainingset/trainset" + Integer.toString(t), true));
                    for(int i=splitSize*t; i<splitSize*(t+1) && i<newset.numInstances(); i++)
                    {
                        for(int j=0; j<newset.numAttributes(); j++)
                        {
                            bw.write(Double.toString(newset.instance(i).value(j)));
                            bw.write(",");
                        }
                        bw.write("\r\n");
                    }

                    bw.close();
                }

                //serialize all objects so we can access them in map functions
                FileOutputStream fos;
		ObjectOutputStream out;
		String filename;

		filename = "affinityFunction.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(affinityFunction);
		out.close();

                filename = "dataset.ser";
		fos = new FileOutputStream(filename);
		out = new ObjectOutputStream(fos);
		out.writeObject(newset.instance(0));
		out.close();


		String inputUri  = "hdfs:///FraudDetection/input/";
		String outputUri = "hdfs:///FraudDetection/AffThresholdOutput/";
		String cacheUri  = "hdfs:///FraudDetection/AffThresholdCache/";


                 //write normalized files in cache
                Process p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./affinityFunction.ser " + cacheUri);
                p.waitFor();

                p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./dataset.ser " + cacheUri);
                p.waitFor();

                //copy input files to input uri
                for(int i=0; i<t; i++)
                {
                    p = Runtime.getRuntime().exec("hadoop fs -copyFromLocal ./trainingset/trainset" + Integer.toString(i) + " " + inputUri);
                    p.waitFor();
                }


                Configuration conf = new Configuration();
                Job job = new Job(conf);

		job.setJarByClass(Utils.class);
		job.setJobName("Compute Affinity Threshold");

		FileInputFormat.addInputPath(job, new Path(inputUri));
                FileOutputFormat.setOutputPath(job, new Path(outputUri));

                job.setMapperClass(AffThresholdMapper.class);
                job.setCombinerClass(AffThresholdCombiner.class);
                job.setReducerClass(AffThresholdReducer.class);

                job.setNumReduceTasks(1);

                job.setOutputKeyClass(IntWritable.class);
                job.setOutputValueClass(Text.class);

                job.setMapOutputKeyClass(IntWritable.class);
                job.setMapOutputValueClass(Text.class);


                DistributedCache.addCacheFile(new URI(cacheUri+"affinityFunction.ser"), job.getConfiguration());
                DistributedCache.addCacheFile(new URI(cacheUri+"dataset.ser"), job.getConfiguration());


		//add variables to the configuration so we can use them in map function
                job.getConfiguration().setInt("dist", dist);
                job.getConfiguration().set("MaxDist", Double.toString(affinityFunction.getMaxDist()));

		job.waitForCompletion(true);


                //copy affinity threshold to the local path
                p = Runtime.getRuntime().exec("hadoop fs -copyToLocal " + outputUri + "/* ./AffThreshold/");
                p.waitFor();

        }


    public final static int performPrunning(
            CellPool aMemoryPool,
            Instances instances,
            AffinityFunction affinityFunction,
            int dist) throws FileNotFoundException, IOException
    {
        LinkedList<Cell> cells = aMemoryPool.getCells();
        int totalPruned = 0;

        // clear usage
        for(Cell c : cells)
        {
            c.clearUsage();
        }

        // calculate usage

        for (int i = 0; i < instances.numInstances(); i++)
        {
            Cell best = aMemoryPool.affinityResponseNormalised(instances.instance(i), affinityFunction, dist).getFirst();
            best.incrementUsage();
        }


        // remove all without usage
        for (Iterator<Cell> iter = cells.iterator(); iter.hasNext();)
        {
            Cell element = iter.next();
            if(element.getUsage() == 0)
            {
                iter.remove();
                totalPruned++;
            }
        }

        return totalPruned;
    }

  	static class AffThresholdMapper extends Mapper<LongWritable, Text, IntWritable, Text>
        {
                public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
                {
                    context.write(new IntWritable(1), value);
                }

        }

        static class AffThresholdCombiner extends Reducer<IntWritable, Text, IntWritable, Text>
        {
            private Instance dataset;
            private AffinityFunction affinityFunction;
            private int dist=1;
            private double MaxDist;


            public void configure(Configuration job) throws IOException, ClassNotFoundException
            {
                        dist = Integer.parseInt(job.get("dist"));
                        MaxDist = Double.valueOf(job.get("MaxDist"));

			//read all serialized files from cache
                        Path[] cacheFiles = new Path[0];
                        cacheFiles = DistributedCache.getLocalCacheFiles(job);

                	FileInputStream fis = null;
                	ObjectInputStream in = null;

                        for (Path cacheFile : cacheFiles)
                        {
                            if((cacheFile.toString()).endsWith("affinityFunction.ser"))
                            {
		        	fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	affinityFunction = (AffinityFunction) in.readObject();
		        	in.close();
                            }

                            else if((cacheFile.toString()).endsWith("dataset.ser"))
                            {
		        	fis = new FileInputStream(cacheFile.toString());
		        	in = new ObjectInputStream(fis);
		        	dataset = (Instance) in.readObject();
		        	in.close();
                            }
                        }
            }


            public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException
            {
                ArrayList<Instance> InstanceList = new ArrayList<Instance>();

                    try {
                        configure(context.getConfiguration());}
                    catch (Exception ex) {
                        throw new RuntimeException("Cannot call configure function: "+ex.getMessage(), ex);}


                    int count = 0;
                    double sumAffinity = 0.0;

                    for(Text value: values)
                    {
                        String s = Text.decode(value.getBytes());

                        String[] sarray = s.split(",");
                        double[] d = new double[sarray.length];

                        for(int i=0; i<dataset.numAttributes(); i++)
                        {
                            try{
                                d[i] = Double.valueOf(sarray[i]);
                                dataset.setValue(i, d[i]);
                            }
                            catch(Exception e)
                            {}
                        }

                        Instance current = dataset;
                        InstanceList.add(current);
                    }


                for (int i = 0; i < InstanceList.size(); i++)
                {
                    for (int j = i+1; j < InstanceList.size(); j++)
                    {
                        double distance = affinityFunction.affinityNormalised(InstanceList.get(i), InstanceList.get(j), dist);
                        sumAffinity += distance;
                        count++;
                    }
                }
                    context.write(new IntWritable(1),
                            new Text(Double.toString(sumAffinity) + "," + Double.toString(count)));
            }
        }
        static class AffThresholdReducer extends Reducer<IntWritable, Text, IntWritable, Text>
	{

		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException
		{
                    double sum = 0;
                    int count = 0;

                    for(Text value: values)
                    {
                        String s = Text.decode(value.getBytes());

                        String[] sarray = s.split(",");
                    }

                    sum /= count;
                    context.write(new IntWritable(1) , new Text("," + Double.toString(sum)));
		}
        }
}
