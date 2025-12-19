# Contributing to Z-Image Distillation Trainer

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (GPU, CUDA version, Python version)
- Error messages or logs

### Suggesting Enhancements

We welcome enhancement suggestions! Please create an issue with:
- Clear description of the proposed feature
- Use case and benefits
- Potential implementation approach (optional)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow the code style (see below)
   - Add tests if applicable
   - Update documentation
4. **Commit your changes**
   ```bash
   git commit -m "Add: your feature description"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**
   - Describe your changes
   - Reference any related issues

## 📝 Code Style

### Python Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and concise

### Code Formatting
We use `black` and `isort` for code formatting:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black *.py
isort *.py
```

### Naming Conventions
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

## 🧪 Testing

Before submitting a PR:

1. **Test your changes locally**
   ```bash
   # Test CLI training
   python z_image_distillation_trainer.py --help
   
   # Test GUI
   python launch_gui.py
   
   # Test with sample data
   python z_image_distillation_trainer.py \
     --train_data_file examples/sample_training_data.json \
     --num_epochs 1 \
     --max_train_prompts 5
   ```

2. **Verify no regressions**
   - Existing functionality still works
   - No new warnings or errors
   - Documentation is updated

## 📚 Documentation

When adding features:
- Update README.md if user-facing
- Update relevant docs/ files
- Add docstrings to new code
- Include usage examples

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Training speed optimizations
- [ ] Memory efficiency improvements
- [ ] Additional loss functions
- [ ] More validation metrics
- [ ] Better error messages

### Medium Priority
- [ ] Additional configuration presets
- [ ] Loss curve visualization in GUI
- [ ] Multi-GPU support
- [ ] Checkpoint management tools
- [ ] Automated testing suite

### Nice to Have
- [ ] Web-based interface
- [ ] Integration with other frameworks
- [ ] Additional example configurations
- [ ] Performance benchmarks
- [ ] Video tutorials

## 🔍 Code Review Process

1. **Automated checks** run on PR creation
2. **Manual review** by maintainers
3. **Discussion and iteration** if needed
4. **Merge** once approved

## 💡 Development Setup

### Initial Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/z-image-distillation-trainer.git
cd z-image-distillation-trainer

# Install in development mode
pip install -e ".[dev]"

# Install diffusers from source (required)
pip install git+https://github.com/huggingface/diffusers
```

### Testing Changes

```bash
# Test CLI
python z_image_distillation_trainer.py --help

# Test GUI
python launch_gui.py

# Run with minimal settings
python z_image_distillation_trainer.py \
  --teacher_model Tongyi-MAI/Z-Image-Base \
  --train_data_file examples/sample_training_data.json \
  --output_dir ./test_output \
  --num_epochs 1 \
  --max_train_prompts 10
```

## 🐛 Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### GPU Debugging

```bash
# Monitor VRAM
nvidia-smi -l 1

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Common Issues

**Import errors**: Install missing dependencies
```bash
pip install -r requirements.txt
```

**CUDA out of memory**: Reduce batch size or enable gradient checkpointing

**Slow training**: Enable Flash Attention, use BF16

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## 🙏 Acknowledgments

Special thanks to:
- Tongyi-MAI team for Z-Image
- All contributors and users
- Open source community

## 📧 Questions?

- Create an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be respectful and constructive

---

Thank you for contributing to Z-Image Distillation Trainer! 🚀
