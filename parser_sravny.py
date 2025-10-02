import requests
import json
import time
import random
from typing import List, Dict

def extract_review_data(review: Dict) -> Dict:
    """Извлекает необходимые поля из отзыва"""
    return {
        "authorName": review.get("authorName"),
        "rating": review.get("rating"),
        "date": review.get("date"),
        "reviewTag": review.get("reviewTag"),
        "text": review.get("text")
    }

def get_reviews_page(page_index: int, page_size: int = 10) -> Dict:
    """Получает одну страницу отзывов"""
    url = f"https://www.sravni.ru/proxy-reviews/reviews?FilterBy=withRates&LocationGarId&NewIds=true&OrderBy=byPopularity&PageIndex={page_index}&PageSize={page_size}&Rated=any&ReviewObjectType=banks&SqueezesVectorIds&Tag=&WithVotes=true&fingerPrint=d337cc677ed5ab6c564599166c436ff4"
    
    headers = {
        "accept": "application/json",
        "accept-language": "ru,en;q=0.9",
        "baggage": "sentry-environment=production,sentry-release=ceba78bf,sentry-public_key=eca1eed372c03cdff0768b2d1069488d,sentry-trace_id=2d6be9168c0b45e6bf3a399982486351,sentry-transaction=%2Flist,sentry-sampled=true,sentry-sample_rand=0.4070634632465757,sentry-sample_rate=1",
        "cookie": ".ASPXANONYMOUS=aB5lmfuUPk2p2GwB6sN2_A; _SLv2_=1405113; _SL_=6.83.; _ym_uid=1749468095187976624; _ym_d=1749468095; _ga=GA1.1.982717873.1749468096; mindboxDeviceUUID=d40f261b-5acf-4f0b-9eae-16d2851a9a4f; directCrm-session=%7B%22deviceGuid%22%3A%22d40f261b-5acf-4f0b-9eae-16d2851a9a4f%22%7D; uxs_uid=e8567f00-4523-11f0-9896-b13387f3399e; _cfuvid=aCC19lYqC.hxJPHO_WpVLfNh46R6rq9hnq3Fvmx5TKY-1758832335782-0.0.1.1-604800000; clid=yclid|4009072402098552831; _gcl_au=1.1.1771390031.1758832339; systemTheme=lager; tmr_lvid=87297870dd0ff60cc3868890e4d5730d; tmr_lvidTS=1758832339448; popmechanic_sbjs_migrations=popmechanic_1418474375998%3D1%7C%7C%7C1471519752600%3D1%7C%7C%7C1471519752605%3D1; __utmz=utmccn%3d(not%20set)%7cutmcsr%3dcloudoff.dit.mos.ru%7cutmcmd%3dreferral%7cutmcct%3d(not%20set)%7cutmctr%3d(not%20set); userTheme=lager; __zzatgib-w-sravniru=MDA0dBA=Fz2+aQ==; __zzatgib-w-sravniru=MDA0dBA=Fz2+aQ==; cfidsgib-w-sravniru=lz6//S3gbPnWqv3F1QJoIvm9QV8ePhQ0siW17GOCzSRposGlwJI6KfvqT/YJBzJI2Jpw4bLKyh8Dv8oe1wRVzBe1rTaJ3QjrVx+2zulE01FlXVkE00uE9+ceRiQCNl+lUwV0ti2o/f1nJoqIc5rGvo38ANMKXskkppCs; cfidsgib-w-sravniru=lz6//S3gbPnWqv3F1QJoIvm9QV8ePhQ0siW17GOCzSRposGlwJI6KfvqT/YJBzJI2Jpw4bLKyh8Dv8oe1wRVzBe1rTaJ3QjrVx+2zulE01FlXVkE00uE9+ceRiQCNl+lUwV0ti2o/f1nJoqIc5rGvo38ANMKXskkppCs; _FP_=d337cc677ed5ab6c564599166c436ff4; _iplv2=1405113; _ipl=6.83.; _ym_isad=2; domain_sid=-ZMyWs0vcF68LhMN-SOrI%3A1759256647990; __cf_bm=cuB4undoLVsfmM3npuA3DSW6qfEmo3YZs1ZdF2As9fE-1759306414-1.0.1.1-q9qmUF3Z55hV9pQ099lT1fUp7DmpzN7GbpxiUMsyT6YeF1gBMsT5uA6TvRmpyfC66ooOWyyxsisphni4N6TgR5YT6DLfzbmxA07qEDZsVttLf3GbLCMFAKorGe4Dhc4o; cf_clearance=BaBPHYlM0uz7DEBgrE53BMrC7jcVZuGRuReuBwTEbGo-1759306444-1.2.1.1-oo5ckDtqFmLZcmdmKike9h34LmA2evSBEmVin2yXsatsNscSaI5Y4RvWDbqpScgJyeZnU9pTMmaJkl4c41.2JJ7I21xdLV68GoaEq9nsbs4KExhpEEAcYDT7W5hJFqmGEW1_2JfJH_.TzZcdx5EHI3gvPFIvzpAhsNafKpS4LyivNJbewu.45SOkZ74AvQIWg6SnhPssFXt9B.0aKt_ztYsfN7iUAWiAxYFmDaiKQqc; tmr_detect=0%7C1759306541616; _ga_WE262B3KPE=GS2.1.s1759306411$o12$g1$t1759306574$j42$l0$h0",
        "if-none-match": 'W/"b261-yVXL9j1TGos5M5+jWkxgKr/lNXs"',
        "priority": "u=1, i",
        "referer": "https://www.sravni.ru/banki/otzyvy/",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "YaBrowser";v="25.8", "Yowser";v="2.5"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sentry-trace": "2d6be9168c0b45e6bf3a399982486351-a53d9745d1ad5dcd-1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 YaBrowser/25.8.0.0 Safari/537.36",
        "x-request-id": "ea9acd03-dfe6-48f7-bf4b-7165f78bb50c",
        "x-requested-with": "XMLHttpRequest"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Ошибка при получении страницы {page_index}: {e}")
        return None

def scrape_all_reviews(exit_p: int = 0, total_reviews: int = 161131, batch_size: int = 1000) -> None:
    """Парсит все отзывы и сохраняет в JSON"""
    all_extracted_reviews = []
    page_size = 10  # Количество отзывов на странице
    total_pages = (total_reviews + page_size - 1) // page_size
    
    print(f"Всего отзывов: {total_reviews}")
    print(f"Всего страниц: {total_pages}")
    
    for page_index in range(exit_p, total_pages):
        try:
            # Случайная задержка между запросами (от 1 до 3 секунд)
            delay = random.uniform(1, 3)
            time.sleep(delay)
            
            print(f"Обрабатывается страница {page_index + 1}/{total_pages}")
            
            data = get_reviews_page(page_index, page_size)
            
            if data and "items" in data:
                page_reviews = data["items"]
                
                for review in page_reviews:
                    extracted_review = extract_review_data(review)
                    all_extracted_reviews.append(extracted_review)
                
                print(f"Страница {page_index + 1} обработана. Получено отзывов: {len(page_reviews)}")
                
                # Сохраняем промежуточные результаты каждые batch_size отзывов
                if len(all_extracted_reviews) % batch_size == 0:
                    save_progress(all_extracted_reviews, len(all_extracted_reviews))
                    
            else:
                print(f"Пустой ответ или ошибка на странице {page_index + 1}")
                
        except Exception as e:
            print(f"Критическая ошибка на странице {page_index + 1}: {e}")
            # Сохраняем прогресс при ошибке
            save_progress(all_extracted_reviews, len(all_extracted_reviews))
            continue
    
    # Финальное сохранение
    save_final_results(all_extracted_reviews)

def save_progress(reviews: List[Dict], count: int) -> None:
    """Сохраняет промежуточные результаты"""
    filename = f"sravni_reviews_progress_{count}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"Промежуточные результаты сохранены в {filename}")

def save_final_results(reviews: List[Dict]) -> None:
    """Сохраняет финальные результаты"""
    filename = "sravni_reviews_final.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"Финальные результаты сохранены в {filename}")
    print(f"Всего собрано отзывов: {len(reviews)}")

def resume_scraping(existing_count: int, total_reviews: int = 161131) -> None:
    """Продолжает парсинг с определенной точки"""
    all_extracted_reviews = []
    
    # Загружаем существующие данные если есть
    try:
        with open(f"sravni_reviews_progress_{existing_count}.json", 'r', encoding='utf-8') as f:
            all_extracted_reviews = json.load(f)
        print(f"Загружено {len(all_extracted_reviews)} существующих отзывов")
    except FileNotFoundError:
        print("Файл с прогрессом не найден, начинаем заново")
        return
    
    page_size = 10
    start_page = existing_count // page_size
    
    # Продолжаем парсинг с нужной страницы
    for page_index in range(start_page, (total_reviews + page_size - 1) // page_size):
        # ... остальной код аналогичен scrape_all_reviews ...
        try:
            # Случайная задержка между запросами (от 1 до 3 секунд)
            delay = random.uniform(1, 3)
            time.sleep(delay)
            
            print(f"Обрабатывается страница {page_index + 1}/{total_reviews}")
            
            data = get_reviews_page(page_index, page_size)
            
            if data and "items" in data:
                page_reviews = data["items"]
                
                for review in page_reviews:
                    extracted_review = extract_review_data(review)
                    all_extracted_reviews.append(extracted_review)
                
                print(f"Страница {page_index + 1} обработана. Получено отзывов: {len(page_reviews)}")
                
                # Сохраняем промежуточные результаты каждые batch_size отзывов
                if len(all_extracted_reviews) % batch_size == 0:
                    save_progress(all_extracted_reviews, len(all_extracted_reviews))
                    
            else:
                print(f"Пустой ответ или ошибка на странице {page_index + 1}")
                
        except Exception as e:
            print(f"Критическая ошибка на странице {page_index + 1}: {e}")
            # Сохраняем прогресс при ошибке
            save_progress(all_extracted_reviews, len(all_extracted_reviews))
            continue
    
    # Финальное сохранение
    save_final_results(all_extracted_reviews)


if __name__ == "__main__":
    # Начать полный парсинг
    scrape_all_reviews(1000)
    
    # Или продолжить с определенной точки (если прервалось)
    # resume_scraping(existing_count=5000)