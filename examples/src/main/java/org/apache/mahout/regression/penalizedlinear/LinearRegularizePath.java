/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.regression.penalizedlinear;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.regression.feature.extractor.FeatureExtractUtility;
import org.apache.mahout.regression.feature.extractor.FeatureExtractorKeySet;
import org.apache.mahout.regression.feature.extractor.FeatureExtractorMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.SortedSet;
import java.util.TreeSet;

public class LinearRegularizePath extends AbstractJob {

  private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";

  private class LinearRegularizePathParameter {
    public int numOfCV;
    public String dependent;
    public String independent;
    public String interaction;
    public float alpha;
    public String lambda;
    public boolean intercept;
  }

  private LinearRegularizePathParameter parameter;
  private String featureNames;
  private String separator;

  private String input;
  private String output;

  private PenalizedLinearSolver solver;

  private LinearRegularizePath() {
  }

  private static final Logger log = LoggerFactory.getLogger(LinearRegularizePath.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new LinearRegularizePath(), args);
  }

  private boolean validateParameter(LinearRegularizePathParameter parameter, String featureNames, String separator) {
    String pattern = FeatureExtractUtility.SeparatorToPattern(separator);
    if (parameter.dependent.equals("") && parameter.interaction.equals("")) {
      log.error("both of the dependent and interaction are empty!");
      return false;
    } else {
      String[] features = featureNames.trim().split(pattern);
      SortedSet<String> featureSet = new TreeSet<String>();
      for (int i = 0; i < features.length; ++i) {
        featureSet.add(features[i]);
      }
      if (!parameter.independent.equals("")) {
        String[] independent = parameter.independent.split(",");
        for (int i = 0; i < independent.length; ++i) {
          if (!featureSet.contains(independent[i])) {
            return false;
          }
        }
      }
      if (!parameter.interaction.equals("")) {
        String[] interaction = parameter.interaction.split(",");
        for (int i = 0; i < interaction.length; ++i) {
          if ((!featureSet.contains(interaction[i].split(":")[0])) || (!featureSet.contains(interaction[i].split(":")[1]))) {
            return false;
          }
        }
      }
      return featureSet.contains(parameter.dependent);
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    if (parseArgs(args)) {
      String[] inputPath = input.split("/");
      String suffix = inputPath[inputPath.length - 1].split("\\.")[1];
      separator = FeatureExtractUtility.ExtensionToSeparator(suffix);
      FileSystem fs = FileSystem.get(getConf());
      BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(input))));
      try {
        featureNames = br.readLine();
      } finally {
        br.close();
      }
      if (!validateParameter(parameter, featureNames, separator)) {
        log.error("feature names provided are not correct!");
        return 1;
      }
      run();
    }
    return 0;
  }

  private void run() throws Exception {
    runFeatureExtractor();
    runPenalizedLinear();
  }

  private void runPenalizedLinear() throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = getConf();
    conf.setInt(PenalizedLinearKeySet.NUM_CV, parameter.numOfCV);
    conf.setFloat(PenalizedLinearKeySet.ALPHA, parameter.alpha);
    conf.set(PenalizedLinearKeySet.LAMBDA, parameter.lambda);
    conf.setBoolean(PenalizedLinearKeySet.INTERCEPT, parameter.intercept);

    Job job = new Job(conf, "Penalized Linear Regression Driver running over input: " + input);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(PenalizedLinearMapper.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setReducerClass(PenalizedLinearReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setCombinerClass(PenalizedLinearReducer.class);
    job.setNumReduceTasks(1);
    job.setJarByClass(LinearRegularizePath.class);

    FileInputFormat.addInputPath(job, new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT));
    FileOutputFormat.setOutputPath(job, new Path(output, "output"));
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("Penalized Linear Regression Job failed processing " + input);
    }
    solver = new PenalizedLinearSolver();
    solver.setAlpha(parameter.alpha);
    solver.setIntercept(parameter.intercept);
    solver.setLambdaString(parameter.lambda);
    solver.initSolver(new Path(output, "output"), getConf());
    solver.regularizePath(solver.getLambda());
    printInfo(parameter, solver);
  }

  private void printInfo(LinearRegularizePathParameter parameter, PenalizedLinearSolver solver)
      throws IOException {
    PenalizedLinearSolver.Coefficients[] coefficients = solver.getCoefficients();
    double[] lambdas = solver.getLambda();
    String model = "model:";
    model += " " + parameter.dependent + " ~";
    if (parameter.intercept) {
      model += " " + "intercept";
    } else {
      model += " " + "0";
    }
    if (!parameter.independent.equals("")) {
      String[] independent = parameter.independent.split(",");
      for (int i = 0; i < independent.length; ++i) {
        model += " + " + independent[i];
      }
    }
    if (!parameter.interaction.equals("")) {
      String[] interaction = parameter.interaction.split(",");
      for (int i = 0; i < interaction.length; ++i) {
        model += " + " + interaction[i];
      }
    }
    System.out.println();
    System.out.println(model);
    System.out.println("The coefficients are in file: " + output + "/coefficients.txt");
    FileSystem fs = FileSystem.get(getConf());
    BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(output, "coefficients.txt"), true)));
    for (int i = 0; i < coefficients.length; ++i) {
      String line = "" + lambdas[i];
      line += " " + coefficients[i].beta0;
      for (int j = 0; j < coefficients[i].beta.length; ++j) {
        line += " " + coefficients[i].beta[j];
      }
      br.write(line + "\n");
    }
    br.close();
  }

  private void runFeatureExtractor() throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set("vector.implementation.class.name", "org.apache.mahout.math.RandomAccessSparseVector");
    conf.set(FeatureExtractorKeySet.FEATURE_NAMES, featureNames);
    conf.set(FeatureExtractorKeySet.SELECTED_DEPENDENT, parameter.dependent);
    conf.set(FeatureExtractorKeySet.SELECTED_INDEPENDENT, parameter.independent);
    conf.set(FeatureExtractorKeySet.SELECTED_INTERACTION, parameter.interaction);
    conf.set(FeatureExtractorKeySet.SEPARATOR, separator);
    Job job = new Job(conf, "Input Driver running over input: " + input);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(FeatureExtractorMapper.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(LinearRegularizePath.class);

    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT));

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFile = builder.withLongName("input")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("where to get training data (CSV or white-spaced TEXT file)")
        .create();

    Option outputFile = builder.withLongName("output")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("where to get results")
        .create();

    Option dependent = builder.withLongName("dependent")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("dependent").withMinimum(1).withMaximum(1).create())
        .withDescription("the dependent features")
        .create();

    Option independent = builder.withLongName("independent")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("independent").create())
        .withDescription("the independent features")
        .create();

    Option interaction = builder.withLongName("interaction")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("interaction").withMinimum(0).create())
        .withDescription("the interactions of features, the format is: feature1:feature2 (identical features are OK)")
        .create();

    Option bias = builder.withLongName("bias")
        .withDescription("include a bias term")
        .create();

    Option lambda = builder.withLongName("lambda")
        .withArgument(argumentBuilder.withName("lambda").withDefault("0").withMinimum(1).create())
        .withDescription("an increasing positive sequence of penalty coefficient, " +
            "with length n >= 0; if lambda is not specified, the sequence is chosen by algorithm.")
        .create();

    Option alpha = builder.withLongName("alpha")
        .withArgument(argumentBuilder.withName("alpha").withDefault("1").withMinimum(1).withMaximum(1).create())
        .withDescription("the elastic-net coefficient with default value 1 (LASSO)")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFile)
        .withOption(outputFile)
        .withOption(dependent)
        .withOption(independent)
        .withOption(interaction)
        .withOption(bias)
        .withOption(lambda)
        .withOption(alpha)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine = parser.parseAndHelp(args);
    if (cmdLine == null) {
      return false;
    }

    parameter = new LinearRegularizePathParameter();
    parameter.numOfCV = 1;
    parameter.alpha = Float.parseFloat((String) cmdLine.getValue(alpha));
    parameter.intercept = cmdLine.hasOption(bias);
    parameter.dependent = (String) cmdLine.getValue(dependent);
    String independentString = "";
    for (Object x : cmdLine.getValues(independent)) {
      independentString += x.toString() + ",";
    }
    parameter.independent = independentString.substring(0, Math.max(independentString.length() - 1, 0));
    String interactionString = "";
    for (Object x : cmdLine.getValues(interaction)) {
      interactionString += x.toString() + ",";
    }
    parameter.interaction = interactionString.substring(0, Math.max(interactionString.length() - 1, 0));

    if (!processLambda(parameter, cmdLine, lambda) ||
        parameter.alpha < 0.0 || parameter.alpha > 1.0 ||
        parameter.numOfCV < 1 || parameter.numOfCV > 20) {
      log.error("please make sure the lambda sequence is positive and increasing, and 0.0 <= alphaValue <= 1.0 and 1 <= numofCV <= 20");
      return false;
    }

    input = (String) cmdLine.getValue(inputFile);
    output = (String) cmdLine.getValue(outputFile);
    return true;
  }

  private boolean processLambda(LinearRegularizePathParameter parameter, CommandLine cmdLine, Option lambda) {
    String lambdaSeq = "";
    double previous = Double.NEGATIVE_INFINITY;
    if (cmdLine.hasOption(lambda)) {
      for (Object x : cmdLine.getValues(lambda)) {
        double number = Double.parseDouble(x.toString());
        if (previous >= number || number < 0) {
          return false;
        }
        lambdaSeq += x.toString() + ",";
        previous = number;
      }
      parameter.lambda = lambdaSeq.substring(0, lambdaSeq.length() - 1);
      return true;
    } else {
      parameter.lambda = "";
      return true;
    }
  }
}
