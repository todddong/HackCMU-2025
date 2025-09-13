# 🤖 GPT-Enhanced Search Setup Guide

This guide explains how to set up the GPT-enhanced search functionality in your Calorie Tracker application.

## 🚀 Quick Setup

### Option 1: Using the Setup Script (Recommended)
```bash
cd /Users/todd/Desktop/HackCMU2025/calorie-tracker/backend
python setup_api_key.py
```

### Option 2: Manual Setup
1. **Get your OpenAI API key** from: https://platform.openai.com/api-keys
2. **Create a `.env` file** in the backend directory:
   ```bash
   echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
   ```
3. **Restart the server** to load the new API key

## 🔧 Configuration

The application uses a configuration system that automatically detects whether you have a real API key or are using the dummy key.

### Current Status
- **Dummy API Key**: `sk-dummy-key-replace-with-real-key-later`
- **Real API Key**: Set via environment variable `OPENAI_API_KEY`

### Configuration File
All settings are managed in `config.py`:
- `ENABLE_GPT_SEARCH`: Enable/disable GPT search features
- `ENABLE_GPT_CALORIE_ESTIMATION`: Enable/disable GPT calorie estimation
- `MOCK_MODE`: Automatically enabled when using dummy API key

## 🧪 Testing the Integration

### Test GPT-Enhanced Search
```bash
curl -X GET "http://localhost:8000/api/search-foods/?query=apple&use_gpt=true"
```

### Test Traditional Search
```bash
curl -X GET "http://localhost:8000/api/search-foods/?query=apple&use_gpt=false"
```

## 🎯 Features

### With Real API Key (GPT-Enhanced)
- ✅ Intelligent query processing
- ✅ Enhanced food descriptions
- ✅ Nutritional insights and highlights
- ✅ Confidence scoring
- ✅ Smart search reasoning

### With Dummy API Key (Mock Mode)
- ✅ Enhanced UI with GPT branding
- ✅ Mock nutritional insights
- ✅ Confidence scoring simulation
- ✅ Search reasoning simulation
- ⚠️ No real AI processing

## 🔄 Switching Between Modes

The application automatically detects your API key configuration:

1. **Dummy Key**: Uses mock GPT features with enhanced UI
2. **Real Key**: Uses actual GPT-4o-mini API calls
3. **No Key**: Falls back to traditional USDA search

## 🛠️ Development

### Adding New Features
1. Update `gpt_search_wrapper.py` for new GPT functionality
2. Update `config.py` for new configuration options
3. Update frontend in `dashboard.html` for new UI elements

### Debugging
- Check server logs for GPT API status
- Use `config.MOCK_MODE` to test without API calls
- Enable `DEBUG=True` in environment for detailed logging

## 📝 API Endpoints

### Search Foods
- **URL**: `/api/search-foods/`
- **Parameters**:
  - `query`: Search term
  - `use_gpt`: Enable GPT enhancement (default: true)
  - `page_size`: Number of results
  - `data_type`: Filter by data type

### Response Format
```json
{
  "query": "apple",
  "results": [
    {
      "fdc_id": "gpt_1",
      "name": "Apple, raw, with skin",
      "description": "🤖 AI-Enhanced: Apple, raw, with skin",
      "nutritional_highlights": ["Low calorie option", "Low fat content"],
      "confidence_score": 0.9,
      "search_reasoning": "Matches search criteria for 'apple'",
      "calories": 52.0,
      "protein": 0.3,
      "carbs": 13.8,
      "fat": 0.2
    }
  ],
  "total_results": 1,
  "search_type": "gpt_enhanced"
}
```

## 🎨 Frontend Integration

The frontend automatically displays GPT-enhanced features:
- 🤖 AI confidence scores
- 💡 Search reasoning
- 🎯 Nutritional highlights
- 🔍 Enhanced descriptions

## 🔐 Security Notes

- Never commit real API keys to version control
- Use environment variables for production
- The dummy key is safe for development and demos
- Real API keys should be kept secure and rotated regularly

## 🆘 Troubleshooting

### Common Issues
1. **"Using mock calorie estimator"**: No real API key configured
2. **"GPT query processing failed"**: API key invalid or network issue
3. **"Search failed"**: Check server logs for detailed error

### Getting Help
- Check server logs for detailed error messages
- Verify API key format (starts with `sk-`)
- Test with `use_gpt=false` to isolate issues
- Ensure OpenAI account has sufficient credits
