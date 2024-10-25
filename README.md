# Video Understanding with Generative AI on AWS Workshop

This hands-on workshop, aimed at developers and solution builders, introduces how to leverage AWS AI Services and foundation models (FMs) through [Amazon Bedrock](https://aws.amazon.com/bedrock/) to implement video understanding use cases. This code goes alongside the self-paced or instructor lead workshop here - https://catalog.us-east-1.prod.workshops.aws/ws-video-understanding-on-aws/en-US

**Please follow the prerequisites listed in the link above or ask your AWS workshop instructor how to get started.**

Within this series of labs, you'll explore some of the most common usage patterns we are seeing with our customers for Video Understanding with Generative AI. We will show techniques for segmenting video into smaller clips and creating enriched contextual metadata for video at different levels of segmentation.

Labs include:

- **01 - Video Time Segmentation** \[Estimated time to complete - 45 mins\]
- **02 - Ad Breaks and Contextual Advertising** \[Estimated time to complete - 30 mins\]
- **02 - Video Summarization** \[Estimated time to complete - 30 mins\]
- **03 - Multi-modal Semantic Search** \[Estimated time to complete - 30 mins\]

## Getting Started

This workshop is presented as a series of **Python notebooks**, which you can run from the environment of your choice:

- For a fully-managed environment with rich AI/ML features, we'd recommend using [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/). To get started quickly, you can refer to the [instructions for domain quick setup](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).
- For a fully-managed but more basic experience, you could instead [create a SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html).
- If you prefer to use your existing (local or other) notebook environment, make sure it has [credentials for calling AWS](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

### Clone and use the notebooks

> ℹ️ **Note:** In SageMaker Studio, you can open a "System Terminal" to run these commands by clicking _File > New > Terminal_

Once your notebook environment is set up, clone this workshop repository into it.

```sh
sudo yum install -y unzip
git clone https://github.com/aws-samples/video-understanding-with-generative-ai-on-aws.git
cd video-understanding-with-generative-ai-on-aws
```

You're now ready to explore the lab notebooks! Start with [00_prerequisites.ipynb](00_prerequisites.ipynb).

## Security

The sample code; software libraries; command line tools; proofs of concept; templates; or other related technology (including any of the foregoing that are provided by our personnel) is provided to you as AWS Content under the AWS Customer Agreement, or the relevant written agreement between you and AWS (whichever applies). You should not use this AWS Content in your production accounts, or on production or other critical data. You are responsible for testing, securing, and optimizing the AWS Content, such as sample code, as appropriate for production grade use based on your specific quality control practices and standards. Deploying AWS Content may incur AWS charges for creating or using AWS chargeable resources, such as running Amazon EC2 instances or using Amazon S3 storage.

## License

See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md)

## Notices

See [NOTICES](NOTICE).
