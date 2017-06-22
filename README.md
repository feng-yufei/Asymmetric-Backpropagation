# Asymmetric-Backpropagation

Back propagation, which is based on chain rule of derivatives, is mathematically sound, but not biologically.  
The gradient, viewed in math, is very natural following the gradient formula. However, when some people want to view
it as biological feedback signal, things become implausible:   

To back propagate the signal in ordinary mathematical way, the weight is involved in the gradient expression. Some people
argue that the since neuron cell connection is not bidirectional, backward feedback can never know how strong a connection
is in the forward path, ie, weight in the forward path should not appear in the gradient function.

## Crazy Asymmetric Backpropagation Solution
Since we cannot use connection weight, some crazy person substitute the weight by a random feedback initialized similarly.  
Relevant Paper are:  
[1] How Important Is Weight Symmetry in Backpropagation? Qianli Liao, Joel Z. Leibo, Tomaso Poggio  
[2] Direct Feedback Alignment Provides Learning in Deep Neural Networks Arild Nøkland

What? NIPS paper on this? MIT Lab on this? It seems that some people really want to play with it.

## Quick Summary Of Simple Math inside

Let 

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
