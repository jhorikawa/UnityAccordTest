using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Networks;
using Accord.Neuro.Learning;
using UnityEngine;
using UnityEngine.UI;

public class AccordPerceptronTest : MonoBehaviour {
	public Image bgImage;
	public GameObject btnObject2;
	public GameObject btnObject3;
	public Text label1;

	#region Training Data
	#region Color Data
	private double[][] inputs = {
		new double[] {214,182,65,1},
		new double[] {173,140,56,1},
		new double[] {64,0,178,1},
		new double[] {184,242,140,1},
		new double[] {145,174,105,1},
	};
	#endregion
	#region Supervised Data
	// Warm Color : { 1 } , Cold Color : { -1 }
	private double[][] outputs = {
		new double[] { 1 },
		new double[] { 1 },
		new double[] { -1 },
		new double[] { -1 },
		new double[] { -1 },
	};
	#endregion
	#endregion

	DeepBeliefNetwork network = null;
    double[] selectedColor = new double[4];


	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}


	/// <summary>
	/// Machine Learning
	/// </summary>
	private void Training()
	{
		// Creating Network
		network = new DeepBeliefNetwork(
			new GaussianFunction(),          // Activation function
			inputsCount: 4,                  // Input degree
			hiddenNeurons: new int[] { 1 }); // Output degree

		// Initialize the network weight with gaussian distribution
		new GaussianWeights(network).Randomize();
		network.UpdateVisibleWeights();

		// Creating DBN Learning Algorithm  
		var teacher = new PerceptronLearning(network);

		// Start learning. Do it for 1000 times.
		for (int i = 0; i < 1000; i++)
			teacher.RunEpoch(inputs, outputs);

		// 重みの更新
		network.UpdateVisibleWeights();
	}

	/// <summary>
	/// Begin Machine Learning
	/// </summary>
	/// <param name="sender"></param>
	/// <param name="e"></param>
	public void Button1_Click()
	{
		Training();

		label1.text = "Training Completed";

		btnObject2.SetActive(true);
		btnObject3.SetActive(true);
	}

	/// <summary>
	/// 色選択
	/// </summary>
	/// <param name="sender"></param>
	/// <param name="e"></param>
	public void Button2_Click()
	{
		Color32 tempColor = new Color32(System.Convert.ToByte(Random.Range(0,255)),System.Convert.ToByte(Random.Range(0,255)),System.Convert.ToByte(Random.Range(0,255)),255);

		selectedColor[0] = System.Convert.ToDouble(tempColor.r);
		selectedColor[1] = System.Convert.ToDouble(tempColor.g);
		selectedColor[2] = System.Convert.ToDouble(tempColor.b);
		selectedColor[3] = 1;

		bgImage.color = tempColor;
		label1.text = tempColor.ToString();
	}

	/// <summary>
	/// Evaluate
	/// </summary>
	/// <param name="sender"></param>
	/// <param name="e"></param>
	public void Button3_Click()
	{
		double[] output = network.Compute(selectedColor);

		//  Get an index with most probability.
		string result = "";
		switch(System.Convert.ToInt32(output[0])){
			case 1:
				result = "Warm Color";
				break;
			case -1:
				result = "Cold Color";
				break;
		}
		//string result = output[0] >= 0 ? "Warm Color" : "Cold Color";

		label1.text = result;
	}
}
