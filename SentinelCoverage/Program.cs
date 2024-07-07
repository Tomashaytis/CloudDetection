using MaxRev.Gdal.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OSGeo.GDAL;
using OSGeo.OSR;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using File = System.IO.File;


namespace SentinelCoverage;

internal partial class Program
{
    private static readonly Dictionary<string, Tuple<double, double, double, double>> Regions = new()
    {
        {"Test1", new Tuple<double, double, double, double>(45.6621,48.9437,46.1070,48.7097)},
        {"Test", new Tuple<double, double, double, double>(44.6621,48.9437,46.1070,47.2097)},
        {"Самарская область", new Tuple<double, double, double, double>(47.8460, 54.6538, 52.7056, 51.7471)},
        {"Пермский край", new Tuple<double, double, double, double>(51.6543, 61.6747, 59.7085, 56.0313)},
        {"Саратовская область", new Tuple<double, double, double, double>(42.3861, 52.9233, 50.9538, 50.2228)},
        {"Амурская область", new Tuple<double, double, double, double>(119.301,57.188,135.698,48.513)},
        {"Астраханская область", new Tuple<double, double, double, double>(44.8621,48.9437,50.1070,45.2097)},
        {"Республика Дагестан", new Tuple<double, double, double, double>(45.0034, 45.0917, 48.9141, 41.0851)},
        {"Волгоградская область", new Tuple<double, double, double, double>(41.0681, 51.3476, 47.5316, 47.3437)},
        {"Оренбургская область", new Tuple<double, double, double, double>(50.6682, 54.4656, 61.7898, 50.3953)},
        {"Ставропольский край", new Tuple<double, double, double, double>(40.7460, 46.1365, 45.8160, 43.7584)}
    };

    private static readonly Dictionary<string, string> UrlTemplates = new()
    {
        {"RGB", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=B04&assets=B03&assets=B02&color_formula=Gamma+RGB+3.2+Saturation+0.8+Sigmoidal+RGB+25+0.35&nodata=0&collection=sentinel-2-l2a&format=png"},
        {"RGB16", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=B04&assets=B03&assets=B02&nodata=0&collection=sentinel-2-l2a&format=tif"},
        {"B08", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=B08&collection=sentinel-2-l2a&format=tif"},
        {"NDVI", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?asset_as_band=true&expression=%28B08-B04%29%2F%28B08%2BB04%29&rescale=-1%2C1&nodata=0&collection=sentinel-2-l2a&format=png"},
        {"LandCover9Classes", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=data&colormap_name=io-lulc-9-class&exitwhenfull=False&skipcovered=False&collection=io-lulc-annual-v02&format=png"},
        {"ESAWorldCover", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=map&colormap_name=esa-worldcover&collection=esa-worldcover&format=png"},
        {"RGBLandsat", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=red&assets=green&assets=blue&color_formula=gamma+RGB+2.7%2C+saturation+1.5%2C+sigmoidal+RGB+15+0.55&nodata=0&collection=landsat-c2-l2&format=png"},
        {"RGB16Landsat", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=red&assets=green&assets=blue&nodata=0&collection=landsat-c2-l2&format=tif"},
        {"NDVILandsat", "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?asset_as_band=true&expression=%28nir08-red%29%2F%28nir08%2Bred%29&nodata=0&rescale=-1%2C1&collection=landsat-c2-l2&format=png"}
    };

    private static Task Main(string[] args)
    {
        GdalBase.ConfigureAll();
        Osr.SetPROJSearchPath("runtimes\\win-x64\\native\\maxrev.gdal.core.libshared");

        using var session = new InferenceSession("UNet_v2.onnx");

        AnsiConsole.Write(
            new FigletText("Planetary Data")
                .LeftJustified()
        .Color(Spectre.Console.Color.Blue));

        AnsiConsole.Write(new Rule("[yellow]Область для загрузки данных[/]").RuleStyle("grey").LeftJustified());
        var region = args.Length == 0 ? AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Область для загрузки")
                .AddChoices(Regions.Keys)) : args[0];
        var lonMin = Regions[region].Item1;
        var latMax = Regions[region].Item2;
        var lonMax = Regions[region].Item3;
        var latMin = Regions[region].Item4;

        var zoom = args.Length == 0 ? AnsiConsole.Ask("Масштабный уровень: ", 13) : int.Parse(args[1]);

        AnsiConsole.Write(new Rule("[yellow]Временной промежуток и облачность[/]").RuleStyle("grey").LeftJustified());
        var startDateString = args.Length == 0 ? AnsiConsole.Ask("Начальная дата: ", "2023-07-01") : args[2];
        var endDateString = args.Length == 0 ? AnsiConsole.Ask("Конечная дата: ", "2023-08-01") : args[3];
        var clouds = args.Length == 0 ? AnsiConsole.Ask("Облачность: ", 100) : int.Parse(args[4]);
        var enableCloudDetection = args.Length == 0 ? AnsiConsole.Confirm("Обнаружение облачности: ", true) : args[5] == "y";
        var enableMaskSaving = args.Length == 0 ? AnsiConsole.Confirm("Сохранение маскок облачности: ", false) : args[5] == "y";

        var startDate = DateTime.Parse(startDateString);
        var endDate = DateTime.Parse(endDateString);

        var dataType = args.Length == 0 ? AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Тип данных")
                .AddChoices(UrlTemplates.Keys)) : args[8];

        AnsiConsole.WriteLine(region);
        AnsiConsole.WriteLine(dataType);

        enableCloudDetection = enableCloudDetection && dataType is "RGB" or "NDVI" or "B08" or "RGB16";
        var cloudPercentLimit = 0.01;
        var dilation = 10;
        var dayStep = 6;
        var maxSimilarTilesCount = 5;
        var correlationLimit = 0.9;
        var dayReserve = 12;

        var urlTemplate = UrlTemplates[dataType];
        if (dataType == "RGB" && startDate.Year < 2022)
            urlTemplate = "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/{0}/WebMercatorQuad/{1}/{2}/{3}@2x?assets=B04&assets=B03&assets=B02&color_formula=Gamma+RGB+3.7+Saturation+1.5+Sigmoidal+RGB+15+0.35&nodata=0&collection=sentinel-2-l2a&format=png";

        var bandsCount = 4;
        if (dataType is "NDVI" or "NDVILandsat")
            bandsCount = 2;
        else if (dataType is "B08")
            bandsCount = 1;

        var planetaryComputerKey = GetPlanetaryComputerKey(dataType, startDate, endDate, clouds).Result;

        (double x, double y) latLon1 = (lonMin, latMax);
        (double x, double y) latLon2 = (lonMax, latMin);

        var tileSize = 512;
        var meters1 = WebMercatorHandler.LatLonToMeters(latLon1.y, latLon1.x);
        var meters2 = WebMercatorHandler.LatLonToMeters(latLon2.y, latLon2.x);
        var xMin = (int)(WebMercatorHandler.FromMetersToPixels(meters1, zoom).x / tileSize);
        var xMax = (int)(WebMercatorHandler.FromMetersToPixels(meters2, zoom).x / tileSize);
        var yMin = (int)Math.Pow(2, zoom) - (int)(WebMercatorHandler.FromMetersToPixels(meters1, zoom).y / tileSize);
        var yMax = (int)Math.Pow(2, zoom) - (int)(WebMercatorHandler.FromMetersToPixels(meters2, zoom).y / tileSize);

        var resHorSize = (xMax - xMin + 1) * tileSize;
        var resVerSize = (yMax - yMin + 1) * tileSize;

        var resFileName = $"{dataType}.tif";

        var startPoint =
            WebMercatorHandler.FromPixelsToMeters((xMin * tileSize, ((int)Math.Pow(2, zoom) - yMin) * tileSize), zoom);
        var resolution = WebMercatorHandler.Resolution(zoom);

        Dataset beforeOutputMask = null;
        Dataset afterOutputMask = null;

        if (dataType is "B08" or "RGB16" or "RGB16Landsat")
        {
            GdalBase.ConfigureAll();
            Gdal.AllRegister();
            var geoTiffDriver = Gdal.GetDriverByName("GTiff");
            File.Delete(resFileName);
            var mosaicRgb = geoTiffDriver.Create(resFileName, resHorSize, resVerSize, bandsCount, DataType.GDT_Int16, null);
            mosaicRgb.SetProjection(
                "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs\"],AUTHORITY[\"EPSG\",\"3857\"]]");
            mosaicRgb.SetGeoTransform([startPoint.x, resolution, 0, startPoint.y, 0, -resolution]);
            mosaicRgb.Dispose();
            var bands = Enumerable.Range(1, bandsCount + 1).ToArray();
            var ext = "tif";

            var processed = 0;
            var total = (xMax - xMin + 1) * (yMax - yMin + 1);

            var failedTiles = new List<Tuple<int, int>>();
            AnsiConsole.Progress()
                .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(),
                    new RemainingTimeColumn(), new SpinnerColumn())
                .Start(ctx =>
                {
                    var downloadTask = ctx.AddTask("[green]Скачиваем данные[/]");
                    downloadTask.MaxValue = total;

                    for (var y = yMin; y <= yMax; y++)
                    {
                        mosaicRgb = Gdal.Open(resFileName, Access.GA_Update);

                        var tiles = new List<int>();
                        for (var x = xMin; x <= xMax; x++)
                        {
                            tiles.Add(x);
                        }

                        var buffer = new int[bandsCount * tileSize * tileSize * (xMax - xMin + 1)];

                        var y1 = y;
                        Parallel.ForEach(tiles, (x, _) =>
                        {
                            var url = string.Format(urlTemplate, planetaryComputerKey, zoom, x, y1);
                            try
                            {
                                var filePath = $"{x}_{y1}.{ext}";
                                DownloadFileAsync(url, filePath).Wait();
                                var tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                var tileData = new int[bandsCount * tileSize * tileSize];
                                tile.ReadRaster(0, 0, tileSize, tileSize, tileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                tile.Dispose();
                                File.Delete(filePath);
                                lock (buffer)
                                {
                                    var shift = (x - xMin) * tileSize;
                                    for (var k = 0; k < bandsCount; k++)
                                        for (var i = 0; i < tileSize; i++)
                                            for (var j = 0; j < tileSize; j++)
                                            {
                                                var index = i * tileSize + j + k * tileSize * tileSize;
                                                var resIndex = i * tileSize * (xMax - xMin + 1) + j + shift + k * tileSize * tileSize * (xMax - xMin + 1);
                                                buffer[resIndex] = tileData[index];
                                            }
                                    processed++;
                                }

                                if (tileData.All(b => b == 0))
                                {
                                    failedTiles.Add(new Tuple<int, int>(x, y1));
                                }
                            }
                            catch (Exception)
                            {
                                lock (buffer)
                                {
                                    processed++;
                                    failedTiles.Add(new Tuple<int, int>(x, y1));
                                }
                            }
                            downloadTask.Value = processed;
                        });

                        mosaicRgb.WriteRaster(0, (y - yMin) * tileSize, tileSize * (xMax - xMin + 1), tileSize, buffer,
                            tileSize * (xMax - xMin + 1),
                            tileSize, bandsCount, bands, 0, 0, 0);
                        mosaicRgb.FlushCache();
                        mosaicRgb.Dispose();
                    }
                });
            if (failedTiles.Count > 0)
            {
                var finallyProcessed = true;
                Environment.Exit(0);
                AnsiConsole.Progress()
                    .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(),
                        new RemainingTimeColumn(), new SpinnerColumn())
                    .Start(ctx =>
                    {
                        processed = 0;
                        var downloadTask = ctx.AddTask("[green]Докачиваем данные[/]");
                        downloadTask.MaxValue = failedTiles.Count;

                        mosaicRgb = Gdal.Open(resFileName, Access.GA_Update);

                        foreach (var tileIndex in failedTiles)
                        {
                            var url = string.Format(urlTemplate, planetaryComputerKey, zoom, tileIndex.Item1, tileIndex.Item2);
                            try
                            {
                                var filePath = $"{tileIndex.Item1}_{tileIndex.Item2}.{ext}";
                                DownloadFileAsync(url, filePath).Wait();
                                var tile = Gdal.Open(filePath, Access.GA_ReadOnly);

                                var tileData = new byte[bandsCount * tileSize * tileSize];
                                tile.ReadRaster(0, 0, tileSize, tileSize, tileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                tile.Dispose();
                                File.Delete(filePath);
                                mosaicRgb.WriteRaster((tileIndex.Item1 - xMin) * tileSize, (tileIndex.Item2 - yMin) * tileSize, tileSize, tileSize, tileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                            }
                            catch (Exception)
                            {
                                finallyProcessed = false;
                            }

                            processed++;
                            downloadTask.Value = processed;
                        }

                        mosaicRgb.FlushCache();
                        mosaicRgb.Dispose();
                    });

                if (!finallyProcessed)
                    AnsiConsole.Write("Есть битые тайлы");
            }
            if (enableCloudDetection)
            {
                var finallyProcessed = true;
                AnsiConsole.Progress()
                .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(),
                   new RemainingTimeColumn(), new SpinnerColumn())
                .Start(ctx =>
                {
                    processed = 0;
                    var downloadTask = ctx.AddTask("[green]Избавляемся от облачности[/]");
                    downloadTask.MaxValue = total;
                    mosaicRgb = Gdal.Open(resFileName, Access.GA_Update);
                    if (enableMaskSaving)
                    {
                        beforeOutputMask = geoTiffDriver.Create("BeforeMask.tif", resHorSize, resVerSize, 4, DataType.GDT_Byte, null);
                        beforeOutputMask.Dispose();
                        beforeOutputMask = Gdal.Open("BeforeMask.tif", Access.GA_Update);
                        afterOutputMask = geoTiffDriver.Create("AfterMask.tif", resHorSize, resVerSize, 4, DataType.GDT_Byte, null);
                        afterOutputMask.Dispose();
                        afterOutputMask = Gdal.Open("AfterMask.tif", Access.GA_Update);
                    }

                    for (var y = yMin; y <= yMax; y++)
                    {
                        var tiles = new List<int>();
                        for (var x = xMin; x <= xMax; x++)
                        {
                            tiles.Add(x);
                        }

                        var y1 = y;
                        _ = Parallel.ForEach(tiles, (x, _) =>
                        {
                            var input = new DenseTensor<float>([1, 3, 512, 512]);
                            var urlRGB16 = string.Format(UrlTemplates["RGB16"], planetaryComputerKey, zoom, x, y1);
                            while (true)
                            {
                                try
                                {
                                    var filePath = $"{x}_{y1}.tif";
                                    DownloadFileAsync(urlRGB16, filePath).Wait();
                                    if (File.Exists(filePath))
                                    {
                                        var tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                        var mainTileDataRGB16 = new int[4 * tileSize * tileSize];
                                        var bandsRGB16 = Enumerable.Range(1, 5).ToArray();
                                        tile.ReadRaster(0, 0, tileSize, tileSize, mainTileDataRGB16, tileSize, tileSize, 4, bandsRGB16, 0, 0, 0);
                                        tile.Dispose();
                                        File.Delete(filePath);
                                        for (var k = 0; k < 3; k++)
                                            for (var i = 0; i < tileSize; i++)
                                                for (var j = 0; j < tileSize; j++)
                                                {
                                                    var index = i * tileSize + j + k * tileSize * tileSize;
                                                    input[0, k, i, j] = Math.Abs(Convert.ToSingle((mainTileDataRGB16[index] - 1175) / 0.25));
                                                }
                                        double cloudPercent;
                                        var mainMask = GetMask(session, tileSize, input, out cloudPercent);
                                        mainMask = Dilate(mainMask, dilation);
                                        if (enableMaskSaving)
                                        {
                                            var tileMask = ConvertMask(mainMask, tileSize);
                                            lock (beforeOutputMask)
                                            {
                                                beforeOutputMask.WriteRaster((x - xMin) * tileSize, (y1 - yMin) * tileSize, tileSize, tileSize, tileMask, tileSize, tileSize, 4, [1, 2, 3, 4], 0, 0, 0);
                                                beforeOutputMask.FlushCache();
                                            }
                                        }
                                        if (cloudPercent > cloudPercentLimit)
                                        {
                                            var tileIndex = new Tuple<int, int>(x, y1);
                                            var mainTileData = new int[bandsCount * tileSize * tileSize];
                                            lock (mosaicRgb)
                                            {
                                                mosaicRgb.ReadRaster((tileIndex.Item1 - xMin) * tileSize, (tileIndex.Item2 - yMin) * tileSize, tileSize, tileSize, mainTileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                            }
                                            for (var k = 0; k < 3; k++)
                                                for (var i = 0; i < tileSize; i++)
                                                    for (var j = 0; j < tileSize; j++)
                                                    {
                                                        var index = i * tileSize + j + k * tileSize * tileSize;
                                                        input[0, k, i, j] = Math.Abs(Convert.ToSingle((mainTileDataRGB16[index] - 1175) / 0.25));
                                                    }

                                            int sameMaskCount = 0;
                                            var tmpStartDate = startDate.AddDays(-dayReserve);
                                            for (DateTime startSearchDate = endDate.AddDays(-dayStep); startSearchDate >= tmpStartDate; startSearchDate = startSearchDate.AddDays(-dayStep))
                                            {
                                                try
                                                {
                                                    DateTime endSearchDate = startSearchDate.AddDays(dayStep);
                                                    var tmpPlanetaryComputerKey = GetPlanetaryComputerKey(dataType, startSearchDate, endSearchDate, 100).Result;
                                                    urlRGB16 = string.Format(UrlTemplates["RGB16"], tmpPlanetaryComputerKey, zoom, tileIndex.Item1, tileIndex.Item2);
                                                    var url = string.Format(urlTemplate, tmpPlanetaryComputerKey, zoom, tileIndex.Item1, tileIndex.Item2);

                                                    filePath = $"{tileIndex.Item1}_{tileIndex.Item2}.tif";
                                                    DownloadFileAsync(urlRGB16, filePath).Wait();
                                                    if (!File.Exists(filePath))
                                                        continue;
                                                    tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                                    var tmpTileDataRGB16 = new int[4 * tileSize * tileSize];
                                                    tile.ReadRaster(0, 0, tileSize, tileSize, tmpTileDataRGB16, tileSize, tileSize, 4, bandsRGB16, 0, 0, 0);
                                                    tile.Dispose();
                                                    File.Delete(filePath);

                                                    filePath = $"{tileIndex.Item1}_{tileIndex.Item2}.png";
                                                    var task = DownloadFileAsync(url, filePath);

                                                    for (var k = 0; k < 3; k++)
                                                        for (var i = 0; i < tileSize; i++)
                                                            for (var j = 0; j < tileSize; j++)
                                                            {
                                                                var index = i * tileSize + j + k * tileSize * tileSize;
                                                                input[0, k, i, j] = Math.Abs(Convert.ToSingle((tmpTileDataRGB16[index] - 1175) / 0.25));
                                                            }
                                                    double tmpCloudPercent;
                                                    byte[,] tmpMask = GetMask(session, tileSize, input, out tmpCloudPercent);
                                                    tmpMask = Dilate(tmpMask, dilation);

                                                    task.Wait();
                                                    if (!File.Exists(filePath))
                                                        continue;
                                                    tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                                    var tmpTileData = new int[bandsCount * tileSize * tileSize];
                                                    tile.ReadRaster(0, 0, tileSize, tileSize, tmpTileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                                    tile.Dispose();
                                                    File.Delete(filePath);

                                                    if (tmpTileDataRGB16.All(b => b == 0))
                                                        continue;
                                                    int alphaChannel = 3 * tileSize * tileSize;
                                                    if (tmpCloudPercent < cloudPercentLimit)
                                                    {
                                                        int cloudPixels = 0;
                                                        for (int i = 0; i < tileSize; i++)
                                                        {
                                                            for (int j = 0; j < tileSize; j++)
                                                            {
                                                                if (tmpTileDataRGB16[i * tileSize + j + alphaChannel] != 0)
                                                                {
                                                                    mainMask[i, j] = tmpMask[i, j];
                                                                    for (int k = 0; k < bandsCount; k++)
                                                                    {
                                                                        var index = i * tileSize + j + k * tileSize * tileSize;
                                                                        mainTileData[index] = tmpTileData[index];
                                                                    }
                                                                }
                                                                if (tmpMask[i, j] is 2 or 3 or 4)
                                                                    cloudPixels++;
                                                            }
                                                        }
                                                        cloudPercent = (double)cloudPixels / (tileSize * tileSize);
                                                    }
                                                    else
                                                    {
                                                        int samePixels = 0;
                                                        int cloudPixels = 0;
                                                        for (int i = 0; i < tileSize; i++)
                                                        {
                                                            for (int j = 0; j < tileSize; j++)
                                                            {
                                                                if (mainMask[i, j] == tmpMask[i, j])
                                                                    samePixels++;
                                                                if (mainMask[i, j] is 2 or 3 or 4)
                                                                {
                                                                    if (tmpTileDataRGB16[i * tileSize + j + alphaChannel] != 0)
                                                                    {
                                                                        if (tmpMask[i, j] is 1)
                                                                        {
                                                                            mainMask[i, j] = tmpMask[i, j];
                                                                            for (int k = 0; k < bandsCount; k++)
                                                                            {
                                                                                var index = i * tileSize + j + k * tileSize * tileSize;
                                                                                mainTileData[index] = tmpTileData[index];
                                                                            }
                                                                        }
                                                                        else if (mainMask[i, j] is 4 && tmpMask[i, j] is 2 or 3)
                                                                        {
                                                                            mainMask[i, j] = tmpMask[i, j];
                                                                            for (int k = 0; k < bandsCount; k++)
                                                                            {
                                                                                var index = i * tileSize + j + k * tileSize * tileSize;
                                                                                mainTileData[index] = tmpTileData[index];
                                                                            }
                                                                            cloudPixels++;
                                                                        }
                                                                        else
                                                                            cloudPixels++;
                                                                    }
                                                                    else
                                                                        cloudPixels++;
                                                                }
                                                            }
                                                        }
                                                        if ((double)samePixels / (tileSize * tileSize) > correlationLimit)
                                                            sameMaskCount++;
                                                        cloudPercent = (double)cloudPixels / (tileSize * tileSize);
                                                    }
                                                    if (sameMaskCount >= maxSimilarTilesCount || cloudPercent < cloudPercentLimit)
                                                        break;
                                                }
                                                catch (Exception) { }
                                            }
                                            lock (mosaicRgb)
                                            {
                                                mosaicRgb.WriteRaster((tileIndex.Item1 - xMin) * tileSize, (tileIndex.Item2 - yMin) * tileSize, tileSize, tileSize, mainTileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                                mosaicRgb.FlushCache();
                                            }
                                        }
                                        if (enableMaskSaving)
                                        {
                                            var tileMask = ConvertMask(mainMask, tileSize);
                                            lock (afterOutputMask)
                                            {
                                                afterOutputMask.WriteRaster((x - xMin) * tileSize, (y1 - yMin) * tileSize, tileSize, tileSize, tileMask, tileSize, tileSize, 4, [1, 2, 3, 4], 0, 0, 0);
                                                afterOutputMask.FlushCache();
                                            }
                                        }
                                        break;
                                    }
                                }
                                catch (Exception) { }
                            }
                            processed++;
                            downloadTask.Value = processed;
                        });
                    }
                    mosaicRgb.Dispose();
                });
                if (!finallyProcessed)
                    AnsiConsole.Write("Облачные тайлы не удалось заменить полностью");
            }
        }
        else
        {
            GdalBase.ConfigureAll();
            Gdal.AllRegister();
            var geoTiffDriver = Gdal.GetDriverByName("GTiff");
            File.Delete(resFileName);
            var mosaicRgb = geoTiffDriver.Create(resFileName, resHorSize, resVerSize, bandsCount, DataType.GDT_Byte, null);
            mosaicRgb.SetProjection(
                "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs\"],AUTHORITY[\"EPSG\",\"3857\"]]");
            mosaicRgb.SetGeoTransform([startPoint.x, resolution, 0, startPoint.y, 0, -resolution]);
            mosaicRgb.Dispose();
            var bands = Enumerable.Range(1, bandsCount + 1).ToArray();
            var ext = "png";

            var processed = 0;
            var total = (xMax - xMin + 1) * (yMax - yMin + 1);

            var failedTiles = new List<Tuple<int, int>>();
            AnsiConsole.Progress()
                .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(),
                    new RemainingTimeColumn(), new SpinnerColumn())
                .Start(ctx =>
                {
                    var downloadTask = ctx.AddTask("[green]Скачиваем данные[/]");
                    downloadTask.MaxValue = total;

                    for (var y = yMin; y <= yMax; y++)
                    {
                        mosaicRgb = Gdal.Open(resFileName, Access.GA_Update);

                        var tiles = new List<int>();
                        for (var x = xMin; x <= xMax; x++)
                        {
                            tiles.Add(x);
                        }

                        var buffer = new byte[bandsCount * tileSize * tileSize * (xMax - xMin + 1)];

                        var y1 = y;
                        Parallel.ForEach(tiles, (x, _) =>
                        {
                            var url = string.Format(urlTemplate, planetaryComputerKey, zoom, x, y1);
                            try
                            {
                                var filePath = $"{x}_{y1}.{ext}";
                                DownloadFileAsync(url, filePath).Wait();
                                if (File.Exists(filePath))
                                {
                                    var tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                    var tileData = new byte[bandsCount * tileSize * tileSize];
                                    tile.ReadRaster(0, 0, tileSize, tileSize, tileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                    tile.Dispose();
                                    File.Delete(filePath);

                                    lock (buffer)
                                    {
                                        var shift = (x - xMin) * tileSize;
                                        for (var k = 0; k < bandsCount; k++)
                                            for (var i = 0; i < tileSize; i++)
                                                for (var j = 0; j < tileSize; j++)
                                                {
                                                    var index = i * tileSize + j + k * tileSize * tileSize;
                                                    var resIndex = i * tileSize * (xMax - xMin + 1) + j + shift + k * tileSize * tileSize * (xMax - xMin + 1);
                                                    buffer[resIndex] = tileData[index];
                                                }
                                        processed++;
                                    }

                                    if (tileData.All(b => b == 0))
                                    {
                                        failedTiles.Add(new Tuple<int, int>(x, y1));
                                    }
                                }
                                else
                                {
                                    processed++;
                                    failedTiles.Add(new Tuple<int, int>(x, y1));
                                }

                            }
                            catch (Exception)
                            {
                                lock (buffer)
                                {
                                    processed++;
                                    failedTiles.Add(new Tuple<int, int>(x, y1));
                                }
                            }
                            downloadTask.Value = processed;
                        });

                        mosaicRgb.WriteRaster(0, (y - yMin) * tileSize, tileSize * (xMax - xMin + 1), tileSize, buffer,
                            tileSize * (xMax - xMin + 1),
                            tileSize, bandsCount, bands, 0, 0, 0);
                        mosaicRgb.FlushCache();
                        mosaicRgb.Dispose();
                    }
                });
            if (failedTiles.Count > 0)
            {
                var finallyProcessed = true;
                AnsiConsole.Progress()
                    .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(),
                        new RemainingTimeColumn(), new SpinnerColumn())
                    .Start(ctx =>
                    {
                        processed = 0;
                        var downloadTask = ctx.AddTask("[green]Докачиваем данные[/]");
                        downloadTask.MaxValue = failedTiles.Count;

                        mosaicRgb = Gdal.Open(resFileName, Access.GA_Update);

                        foreach (var tileIndex in failedTiles)
                        {
                            var url = string.Format(urlTemplate, planetaryComputerKey, zoom, tileIndex.Item1, tileIndex.Item2);
                            try
                            {
                                var filePath = $"{tileIndex.Item1}_{tileIndex.Item2}.{ext}";
                                DownloadFileAsync(url, filePath).Wait();
                                if (File.Exists(filePath))
                                {
                                    var tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                    var tileData = new byte[bandsCount * tileSize * tileSize];
                                    tile.ReadRaster(0, 0, tileSize, tileSize, tileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                    tile.Dispose();
                                    File.Delete(filePath);

                                    mosaicRgb.WriteRaster((tileIndex.Item1 - xMin) * tileSize, (tileIndex.Item2 - yMin) * tileSize, tileSize, tileSize, tileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                }
                            }
                            catch (Exception)
                            {
                                finallyProcessed = false;
                            }

                            processed++;
                            downloadTask.Value = processed;
                        }

                        mosaicRgb.FlushCache();
                        mosaicRgb.Dispose();
                    });
                if (!finallyProcessed)
                    AnsiConsole.Write("Есть битые тайлы");
            }
            if (enableCloudDetection)
            {
                var finallyProcessed = true;
                AnsiConsole.Progress()
                .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(),
                   new RemainingTimeColumn(), new SpinnerColumn())
                .Start(ctx =>
                {
                    processed = 0;
                    var downloadTask = ctx.AddTask("[green]Избавляемся от облачности[/]");
                    downloadTask.MaxValue = total;
                    mosaicRgb = Gdal.Open(resFileName, Access.GA_Update);
                    if (enableMaskSaving)
                    {
                        beforeOutputMask = geoTiffDriver.Create("BeforeMask.tif", resHorSize, resVerSize, 4, DataType.GDT_Byte, null);
                        beforeOutputMask.Dispose();
                        beforeOutputMask = Gdal.Open("BeforeMask.tif", Access.GA_Update);
                        afterOutputMask = geoTiffDriver.Create("AfterMask.tif", resHorSize, resVerSize, 4, DataType.GDT_Byte, null);
                        afterOutputMask.Dispose();
                        afterOutputMask = Gdal.Open("AfterMask.tif", Access.GA_Update);
                    }

                    for (var y = yMin; y <= yMax; y++)
                    {
                        var tiles = new List<int>();
                        for (var x = xMin; x <= xMax; x++)
                        {
                            tiles.Add(x);
                        }

                        var y1 = y;
                        _ = Parallel.ForEach(tiles, (x, _) =>
                        {
                            var input = new DenseTensor<float>([1, 3, 512, 512]);
                            var urlRGB16 = string.Format(UrlTemplates["RGB16"], planetaryComputerKey, zoom, x, y1);
                            while (true)
                            {
                                try
                                {
                                    var filePath = $"{x}_{y1}.tif";
                                    DownloadFileAsync(urlRGB16, filePath).Wait();
                                    if (File.Exists(filePath))
                                    {
                                        var tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                        var mainTileDataRGB16 = new int[4 * tileSize * tileSize];
                                        var bandsRGB16 = Enumerable.Range(1, 5).ToArray();
                                        tile.ReadRaster(0, 0, tileSize, tileSize, mainTileDataRGB16, tileSize, tileSize, 4, bandsRGB16, 0, 0, 0);
                                        tile.Dispose();
                                        File.Delete(filePath);
                                        for (var k = 0; k < 3; k++)
                                            for (var i = 0; i < tileSize; i++)
                                                for (var j = 0; j < tileSize; j++)
                                                {
                                                    var index = i * tileSize + j + k * tileSize * tileSize;
                                                    input[0, k, i, j] = Math.Abs(Convert.ToSingle((mainTileDataRGB16[index] - 1175) / 0.25));
                                                }
                                        double cloudPercent;
                                        var mainMask = GetMask(session, tileSize, input, out cloudPercent);
                                        mainMask = Dilate(mainMask, dilation);
                                        if (enableMaskSaving)
                                        {
                                            var tileMask = ConvertMask(mainMask, tileSize);
                                            lock (beforeOutputMask)
                                            {
                                                beforeOutputMask.WriteRaster((x - xMin) * tileSize, (y1 - yMin) * tileSize, tileSize, tileSize, tileMask, tileSize, tileSize, 4, [1, 2, 3, 4], 0, 0, 0);
                                                beforeOutputMask.FlushCache();
                                            }
                                        }
                                        if (cloudPercent > cloudPercentLimit)
                                        {
                                            var tileIndex = new Tuple<int, int>(x, y1);
                                            var mainTileData = new byte[bandsCount * tileSize * tileSize];
                                            lock (mosaicRgb)
                                            {
                                                mosaicRgb.ReadRaster((tileIndex.Item1 - xMin) * tileSize, (tileIndex.Item2 - yMin) * tileSize, tileSize, tileSize, mainTileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                            }
                                            for (var k = 0; k < 3; k++)
                                                for (var i = 0; i < tileSize; i++)
                                                    for (var j = 0; j < tileSize; j++)
                                                    {
                                                        var index = i * tileSize + j + k * tileSize * tileSize;
                                                        input[0, k, i, j] = Math.Abs(Convert.ToSingle((mainTileDataRGB16[index] - 1175) / 0.25));
                                                    }

                                            int sameMaskCount = 0;
                                            var tmpStartDate = startDate.AddDays(-dayReserve);
                                            for (DateTime startSearchDate = endDate.AddDays(-dayStep); startSearchDate >= tmpStartDate; startSearchDate = startSearchDate.AddDays(-dayStep))
                                            {
                                                try
                                                {
                                                    DateTime endSearchDate = startSearchDate.AddDays(dayStep);
                                                    var tmpPlanetaryComputerKey = GetPlanetaryComputerKey(dataType, startSearchDate, endSearchDate, 100).Result;
                                                    urlRGB16 = string.Format(UrlTemplates["RGB16"], tmpPlanetaryComputerKey, zoom, tileIndex.Item1, tileIndex.Item2);
                                                    var url = string.Format(urlTemplate, tmpPlanetaryComputerKey, zoom, tileIndex.Item1, tileIndex.Item2);

                                                    filePath = $"{tileIndex.Item1}_{tileIndex.Item2}.tif";
                                                    DownloadFileAsync(urlRGB16, filePath).Wait();
                                                    if (!File.Exists(filePath))
                                                        continue;
                                                    tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                                    var tmpTileDataRGB16 = new int[4 * tileSize * tileSize];
                                                    tile.ReadRaster(0, 0, tileSize, tileSize, tmpTileDataRGB16, tileSize, tileSize, 4, bandsRGB16, 0, 0, 0);
                                                    tile.Dispose();
                                                    File.Delete(filePath);

                                                    filePath = $"{tileIndex.Item1}_{tileIndex.Item2}.png";
                                                    var task = DownloadFileAsync(url, filePath);

                                                    for (var k = 0; k < 3; k++)
                                                        for (var i = 0; i < tileSize; i++)
                                                            for (var j = 0; j < tileSize; j++)
                                                            {
                                                                var index = i * tileSize + j + k * tileSize * tileSize;
                                                                input[0, k, i, j] = Math.Abs(Convert.ToSingle((tmpTileDataRGB16[index] - 1175) / 0.25));
                                                            }
                                                    double tmpCloudPercent;
                                                    byte[,] tmpMask = GetMask(session, tileSize, input, out tmpCloudPercent);
                                                    tmpMask = Dilate(tmpMask, dilation);

                                                    task.Wait();
                                                    if (!File.Exists(filePath))
                                                        continue;
                                                    tile = Gdal.Open(filePath, Access.GA_ReadOnly);
                                                    var tmpTileData = new byte[bandsCount * tileSize * tileSize];
                                                    tile.ReadRaster(0, 0, tileSize, tileSize, tmpTileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                                    tile.Dispose();
                                                    File.Delete(filePath);

                                                    if (tmpTileDataRGB16.All(b => b == 0))
                                                        continue;
                                                    int alphaChannel = 3 * tileSize * tileSize;
                                                    if (tmpCloudPercent < cloudPercentLimit)
                                                    {
                                                        int cloudPixels = 0;
                                                        for (int i = 0; i < tileSize; i++)
                                                        {
                                                            for (int j = 0; j < tileSize; j++)
                                                            {
                                                                if (tmpTileDataRGB16[i * tileSize + j + alphaChannel] != 0)
                                                                {
                                                                    mainMask[i, j] = tmpMask[i, j];
                                                                    for (int k = 0; k < bandsCount; k++)
                                                                    {
                                                                        var index = i * tileSize + j + k * tileSize * tileSize;
                                                                        mainTileData[index] = tmpTileData[index];
                                                                    }
                                                                }
                                                                if (tmpMask[i, j] is 2 or 3 or 4)
                                                                    cloudPixels++;
                                                            }
                                                        }
                                                        cloudPercent = (double)cloudPixels / (tileSize * tileSize);
                                                    }
                                                    else
                                                    {
                                                        int samePixels = 0;
                                                        int cloudPixels = 0;
                                                        for (int i = 0; i < tileSize; i++)
                                                        {
                                                            for (int j = 0; j < tileSize; j++)
                                                            {
                                                                if (mainMask[i, j] == tmpMask[i, j])
                                                                    samePixels++;
                                                                if (mainMask[i, j] is 2 or 3 or 4)
                                                                {
                                                                    if (tmpTileDataRGB16[i * tileSize + j + alphaChannel] != 0)
                                                                    {
                                                                        if (tmpMask[i, j] is 1)
                                                                        {
                                                                            mainMask[i, j] = tmpMask[i, j];
                                                                            for (int k = 0; k < bandsCount; k++)
                                                                            {
                                                                                var index = i * tileSize + j + k * tileSize * tileSize;
                                                                                mainTileData[index] = tmpTileData[index];
                                                                            }
                                                                        }
                                                                        else if (mainMask[i, j] is 4 && tmpMask[i, j] is 2 or 3)
                                                                        {
                                                                            mainMask[i, j] = tmpMask[i, j];
                                                                            for (int k = 0; k < bandsCount; k++)
                                                                            {
                                                                                var index = i * tileSize + j + k * tileSize * tileSize;
                                                                                mainTileData[index] = tmpTileData[index];
                                                                            }
                                                                            cloudPixels++;
                                                                        }
                                                                        else
                                                                            cloudPixels++;
                                                                    }
                                                                    else
                                                                        cloudPixels++;
                                                                }
                                                            }
                                                        }
                                                        if ((double)samePixels / (tileSize * tileSize) > correlationLimit)
                                                            sameMaskCount++;
                                                        cloudPercent = (double)cloudPixels / (tileSize * tileSize);
                                                    }
                                                    if (sameMaskCount >= maxSimilarTilesCount || cloudPercent < cloudPercentLimit)
                                                        break;
                                                }
                                                catch (Exception) { }
                                            }
                                            lock (mosaicRgb)
                                            {
                                                mosaicRgb.WriteRaster((tileIndex.Item1 - xMin) * tileSize, (tileIndex.Item2 - yMin) * tileSize, tileSize, tileSize, mainTileData, tileSize, tileSize, bandsCount, bands, 0, 0, 0);
                                                mosaicRgb.FlushCache();
                                            }
                                        }
                                        if (enableMaskSaving)
                                        {
                                            var tileMask = ConvertMask(mainMask, tileSize);
                                            lock (afterOutputMask)
                                            {
                                                afterOutputMask.WriteRaster((x - xMin) * tileSize, (y1 - yMin) * tileSize, tileSize, tileSize, tileMask, tileSize, tileSize, 4, [1, 2, 3, 4], 0, 0, 0);
                                                afterOutputMask.FlushCache();
                                            }
                                        }
                                        break;
                                    }
                                }
                                catch (Exception) { }
                            }
                            processed++;
                            downloadTask.Value = processed;
                        });
                    }
                    mosaicRgb.Dispose();
                });
                if (!finallyProcessed)
                    AnsiConsole.Write("Облачные тайлы не удалось заменить полностью");
            }
        }

        if (enableMaskSaving)
        {
            beforeOutputMask.Dispose();
            afterOutputMask.Dispose();
        }
        return Task.CompletedTask;
    }

    private static async Task DownloadFileAsync(string url, string fileName)
    {
        using var client = new HttpClient();
        var response = await client.GetAsync(url);
        if (response.IsSuccessStatusCode)
        {
            await using var fileStream = File.Create(fileName);
            await response.Content.CopyToAsync(fileStream);
        }
    }

    private static async Task<string> GetPlanetaryComputerKey(string dataType, DateTime start, DateTime end, int clouds)
    {
        var httpClient = new HttpClient();
        var requestBody = GetRequestBody(dataType, $"{start.Year}-{start.Month}-{start.Day}", $"{end.Year}-{end.Month}-{end.Day}", clouds);
        var response = await httpClient.PostAsync("https://planetarycomputer.microsoft.com/api/data/v1/mosaic/register", new StringContent(requestBody, Encoding.UTF8, "application/json"));
        var json = await response.Content.ReadAsStringAsync();
        var rx = PlanetaryComputerKeyRegex();
        var match = rx.Match(json);
        var planetaryComputerKey = match.Groups["key"].Value;
        return planetaryComputerKey;
    }

    private static string GetRequestBody(string dataType, string startDate, string endDate, int clouds)
    {
        return dataType switch
        {
            "RGB" or "NDVI" or "B08" or "RGB16" => $$"""
                                          {
                                              "filter-lang": "cql2-json",
                                              "filter": {
                                                  "op": "and",
                                                  "args": [
                                                      {
                                                          "op": "=",
                                                          "args": [
                                                              {
                                                                  "property": "collection"
                                                              },
                                                              "sentinel-2-l2a"
                                                          ]
                                                      },
                                                      {
                                                          "op": "anyinteracts",
                                                          "args": [
                                                              {
                                                                  "property": "datetime"
                                                              },
                                                              {
                                                                  "interval": [
                                                                      "{{startDate}}",
                                                                      "{{endDate}}T23:59:59Z"
                                                                  ]
                                                              }
                                                          ]
                                                      },
                                                      {
                                                          "op": "<=",
                                                          "args": [
                                                              {
                                                                  "property": "eo:cloud_cover"
                                                              },
                                                              {{clouds}}
                                                          ]
                                                      }
                                                  ]
                                              },
                                              "sortby": [
                                                  {
                                                      "field": "eo:cloud_cover",
                                                      "direction": "asc"
                                                  },
                                                  {
                                                      "field": "datetime",
                                                      "direction": "desc"
                                                  }
                                              ]
                                          }
                                          """,
            "LandCover9Classes" => $$"""
                             {
                                 "filter-lang": "cql2-json",
                                 "filter": {
                                     "op": "and",
                                     "args": [
                                         {
                                             "op": "=",
                                             "args": [
                                                 {
                                                     "property": "collection"
                                                 },
                                                 "io-lulc-annual-v02"
                                             ]
                                         },
                                         {
                                             "op": "anyinteracts",
                                             "args": [
                                                 {
                                                     "property": "datetime"
                                                 },
                                                 {
                                                     "interval": [
                                                         "{{startDate}}",
                                                         "{{endDate}}T23:59:59Z"
                                                     ]
                                                 }
                                             ]
                                         }
                                     ]
                                 },
                                 "sortby": [
                                     {
                                         "field": "datetime",
                                         "direction": "desc"
                                     }
                                 ]
                             }
                             """,
            "ESAWorldCover" => $$"""
                             {
                                 "filter-lang": "cql2-json",
                                 "filter": {
                                     "op": "and",
                                     "args": [
                                         {
                                             "op": "=",
                                             "args": [
                                                 {
                                                     "property": "collection"
                                                 },
                                                 "esa-worldcover"
                                             ]
                                         },
                                         {
                                             "op": "anyinteracts",
                                             "args": [
                                                 {
                                                     "property": "datetime"
                                                 },
                                                 {
                                                     "interval": [
                                                         "{{startDate}}",
                                                         "{{endDate}}T23:59:59Z"
                                                     ]
                                                 }
                                             ]
                                         }
                                     ]
                                 },
                                 "sortby": [
                                     {
                                         "field": "datetime",
                                         "direction": "desc"
                                     }
                                 ]
                             }
                             """,
            _ => $$"""
                   {
                       "filter-lang": "cql2-json",
                       "filter": {
                           "op": "and",
                           "args": [
                               {
                                   "op": "=",
                                   "args": [
                                       {
                                           "property": "collection"
                                       },
                                       "landsat-c2-l2"
                                   ]
                               },
                               {
                                   "op": "anyinteracts",
                                   "args": [
                                       {
                                           "property": "datetime"
                                       },
                                       {
                                           "interval": [
                                               "{{startDate}}",
                                               "{{endDate}}T23:59:59Z"
                                           ]
                                       }
                                   ]
                               },
                               {
                                   "op": "<=",
                                   "args": [
                                       {
                                           "property": "eo:cloud_cover"
                                       },
                                       {{clouds}}
                                   ]
                               }
                           ]
                       },
                       "sortby": [
                           {
                               "field": "eo:cloud_cover",
                               "direction": "asc"
                           },
                           {
                               "field": "datetime",
                               "direction": "desc"
                           }
                       ]
                   }
                   """
        };
    }

    private static byte FindMax(float[] classes)
    {
        byte maxIndex = 0;
        float max = 0;
        for (int i = 0; i < classes.Length; i++)
        {
            if (classes[i] > max)
            {
                max = classes[i];
                maxIndex = Convert.ToByte(i);
            }
        }
        return maxIndex;
    }

    private static byte[,] GetMask(InferenceSession session, int tileSize, DenseTensor<float> input, out double cloudPercent)
    {
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(
        [
            NamedOnnxValue.CreateFromTensor("x.1", input)
        ]);
        if (results.FirstOrDefault()?.Value is not Tensor<float> output)
            throw new ApplicationException("Изображение не может быть обработано");

        cloudPercent = 0;
        var mask = new byte[tileSize, tileSize];
        for (var i = 0; i < tileSize; i++)
        {
            for (var j = 0; j < tileSize; j++)
            {
                byte maskClass1 = FindMax([output[0, 0, i, j], output[0, 1, i, j], output[0, 2, i, j], output[0, 3, i, j], output[0, 4, i, j]]);
                if (maskClass1 is 4 or 3 or 2)
                {
                    mask[i, j] = maskClass1;
                    cloudPercent++;
                }
                else
                    mask[i, j] = 1;
            }
        }
        cloudPercent /= (tileSize * tileSize);
        return mask;
    }

    private static byte[,] Dilate(byte[,] mask, int size)
    {
        byte[,] tmpMask = mask;
        byte[,] newMask = new byte[mask.GetLength(0), mask.GetLength(1)];
        Random TempRandom = new Random();
        int apetureMin = -(size / 2);
        int apetureMax = (size / 2);
        for (int x = 0; x < mask.GetLength(0); ++x)
        {
            for (int y = 0; y < mask.GetLength(1); ++y)
            {
                byte value = 0;
                for (int x2 = apetureMin; x2 < apetureMax; ++x2)
                {
                    int tmpX = x + x2;
                    if (tmpX >= 0 && tmpX < mask.GetLength(0))
                    {
                        for (int y2 = apetureMin; y2 < apetureMax; ++y2)
                        {
                            int tmpY = y + y2;
                            if (tmpY >= 0 && tmpY < mask.GetLength(1))
                            {
                                if (tmpMask[tmpX, tmpY] > value)
                                    value = tmpMask[tmpX, tmpY];
                            }
                        }
                    }
                }
                newMask[x, y] = value;
            }
        }
        return newMask;
    }

    private static byte[] ConvertMask(byte[,] mask, int tileSize)
    {
        var result = new byte[4 * tileSize * tileSize];
        for (var i = 0; i < tileSize; i++)
            for (var j = 0; j < tileSize; j++)
            {
                var indexR = i * tileSize + j;
                var indexG = i * tileSize + j + 1 * tileSize * tileSize;
                var indexB = i * tileSize + j + 2 * tileSize * tileSize;
                var indexA = i * tileSize + j + 3 * tileSize * tileSize;
                result[indexA] = 255;
                switch (mask[i, j])
                {
                    case 0:
                        result[indexR] = 0;
                        result[indexG] = 0;
                        result[indexB] = 0;
                        break;
                    case 1:
                        result[indexR] = 102;
                        result[indexG] = 255;
                        result[indexB] = 255;
                        break;
                    case 2:
                        result[indexR] = 0;
                        result[indexG] = 51;
                        result[indexB] = 51;
                        break;
                    case 3:
                        result[indexR] = 153;
                        result[indexG] = 255;
                        result[indexB] = 51;
                        break;
                    case 4:
                        result[indexR] = 255;
                        result[indexG] = 153;
                        result[indexB] = 0;
                        break;
                }
            }
        return result;
    }

    [GeneratedRegex("mosaic/(?<key>\\w+)/info", RegexOptions.IgnoreCase | RegexOptions.Compiled, "ru-RU")]
    private static partial Regex PlanetaryComputerKeyRegex();

}