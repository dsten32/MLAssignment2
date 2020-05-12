package weka.filters.unsupervised.instance;
import jdk.internal.org.objectweb.asm.tree.TryCatchBlockNode;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import org.apache.commons.math3.distribution.BetaDistribution;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

import java.util.*;
import java.util.stream.Collectors;

public class Smear extends SimpleBatchFilter implements Randomizable, OptionHandler {

    /**
     * The seed for the random number generator.
     */
    protected int m_seed = 0;

    /**
     * The number of instances to output per input instance.
     */
    protected int numSamples = 2;

    /**
     * The value of alpha to use in the beta distribution
     */
    protected double m_stdDev = 0.05;

    /**
     * The number to use for determining the kth smallest attribute difference
     */
    protected int k_gap = 10;

    @Override
    public String globalInfo() {
        return "This filter implements the 'Smear' approach to generate variable data from an initial dataset";
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        if (m_Debug) {
            System.err.println("Determining output format.");
        }
        Instances output = new Instances(inputFormat, 0);
//        output.setClassIndex(output.numAttributes() - 1);
        if (m_Debug) {
            System.err.println("Finished determining output format with " + output.numAttributes() + " attributes.");
        }
        return output;
    }

    @Override
    protected Instances process(Instances instances) throws Exception {
        Random rand = new Random(m_seed);
        double gaussian;
        Instances processedInstances = getOutputFormat();

        // for each numSamples copy incoming instances to new collection.
        for (int sample = 0; sample < numSamples; sample++) {
            processedInstances.addAll(instances);
        }
        double[] attributeGaps = getGaps(instances);

        // perturb the values and update output instances collection.
        if (m_Debug) {
            System.err.println("Gaps array: " + Arrays.toString(attributeGaps));
        }
        for(int i=0;i<attributeGaps.length;i++) {

            for (int j = 0; j < processedInstances.size(); j++) {
                gaussian = rand.nextGaussian() * (m_stdDev * attributeGaps[i]);
                Instance updatedInstance = processedInstances.instance(j);
                updatedInstance.setValue(i, updatedInstance.value(i) + gaussian);
                processedInstances.set(j, updatedInstance);
            }
        }
        return processedInstances;
    }

    private double[] getGaps(Instances instances){
        double[] kGaps = new double[instances.numAttributes()-1];
        double attributeKthGap = 1;
        for (int i = 0; i < instances.numAttributes()-1; i++) {
            // Get instances attribute values
            double[] initialAttributeValues = instances.attributeToDoubleArray(i);
            // Sort values and remove duplicates with TreeSet
            TreeSet<Double> sortedAttributeValues = Arrays.stream(initialAttributeValues).boxed().collect(Collectors.toCollection(TreeSet::new));
            // Set up new TreeSet to hold sorted differences
            SortedSet<Double> sortedDifferences = new TreeSet<>();
            if(m_Debug){
                System.err.println("Sorted attribute values array: " + Arrays.toString(sortedAttributeValues.toArray()));
            }
            // get number of distinct attributes and loop through, adding the difference between
            // consecutive values to differences set
            int numSortedAttributeVals = sortedAttributeValues.size();
            // store calculated difference in temp variable, if the final highest value is NaN then
            // do not calculate a new difference, use the stored difference
            double tempDifference=0;
            for (int j = 0; j < numSortedAttributeVals - 1; j++) {
                double lowestRemaining = sortedAttributeValues.pollFirst();
                double nextLowestRemaining = sortedAttributeValues.higher(lowestRemaining);
                if (((Double)lowestRemaining).isNaN() || ((Double)nextLowestRemaining).isNaN()){
                    sortedDifferences.add(tempDifference);
                    break;}
                tempDifference = Math.abs(nextLowestRemaining - lowestRemaining);
                sortedDifferences.add(tempDifference);
            }
            if(m_Debug) {
                System.err.println("differences array: " + Arrays.toString(sortedDifferences.toArray()));
            }
            // Get kth smallest difference from Set, after turning into array
            Double[] differencesArray = sortedDifferences.toArray(new Double[0]);
            if (differencesArray.length >= k_gap) {
                attributeKthGap = differencesArray[k_gap - 1];
            } else {
                attributeKthGap = differencesArray[differencesArray.length - 1];
            }
            if(m_Debug) {
                System.err.println("kth-gap is: " + attributeKthGap);
            }
            kGaps[i] = attributeKthGap;
    }
        return kGaps;
}

    @OptionMetadata(
            displayName = "Random Seed",
            description = "The seed value for the random number generator.",
            displayOrder = 3,
            commandLineParamName = "S",
            commandLineParamSynopsis = "-S")
    @Override
    public void setSeed(int seed) {
        this.m_seed = seed;
    }

    @Override
    public int getSeed() {
        return m_seed;
    }

    @OptionMetadata(
            displayName = "kgap",
            description = "Value for the 'kth' smallest attribute difference used for scaling the gaussian value",
            displayOrder = 2,
            commandLineParamName = "kgap",
            commandLineParamSynopsis = "-kgap")
    public int getk_gap() {
        return k_gap;
    }

    public void setk_gap(int k_gap) {
        this.k_gap = k_gap;
    }

    @OptionMetadata(
            displayName = "Number of Samples",
            description = "The number of instances to create per instance input.",
            displayOrder = 4,
            commandLineParamName = "numSamples",
            commandLineParamSynopsis = "-numSamples")
    public int getNumSamples() {
        return numSamples;
    }

    public void setNumSamples(int numSamples) {
        this.numSamples = numSamples;
    }

    @OptionMetadata(
            displayName = "std dev",
            description = "The standard deviation used for generating the gaussian smear value.",
            displayOrder = 5,
            commandLineParamName = "stdDev",
            commandLineParamSynopsis = "-stdDev")
    public double getStdDev() {
        return m_stdDev;
    }

    public void setStdDev(double m_stdDev) {
        this.m_stdDev = m_stdDev;
    }


    /**
     * The main method used for running this filter from the command-line interface.
     *
     * @param options the command-line options
     */
    public static void main(String[] options) {
        runFilter(new Smear(), options);
    }
}
