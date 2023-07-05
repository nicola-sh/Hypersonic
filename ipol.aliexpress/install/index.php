<?php
IncludeModuleLangFile(__FILE__);

class ipol_aliexpress extends CModule
{
	var $MODULE_ID = 'ipol.aliexpress';

	var $MODULE_NAME;
	var $MODULE_DESCRIPTION;
	var $MODULE_VERSION;
	var $MODULE_VERSION_DATE;

	var $PARTNER_NAME = 'Ipol';
	var $PARTNER_URI = 'http://www.ipolh.com';

	/**
	 * Конструктор класса
	 */
	public function __construct()
	{
		$arModuleVersion = array();
		include(__DIR__ .'/version.php');

		$this->MODULE_VERSION = $arModuleVersion['VERSION'];
		$this->MODULE_VERSION_DATE = $arModuleVersion['VERSION_DATE'];

		$this->MODULE_NAME = GetMessage($this->MODULE_ID .'_MODULE_NAME');
		$this->MODULE_DESCRIPTION = GetMessage($this->MODULE_ID .'_MODULE_DESCRIPTION');

		$this->PARTNER_NAME = 'Ipol';
		$this->PARTNER_URI = 'http://www.ipolh.com';
	}

	/**
	 * Процесс установки модуля
	 */
	public function DoInstall()
	{
		if (($error = $this->checkDependences()) !== true) {
			$GLOBALS['IPOL_DPD_INSTALL_ERROR'] = $error;
			$GLOBALS['APPLICATION']->IncludeAdminFile(GetMessage($this->MODULE_ID . '_INSTALL_ERROR_TITLE'), __DIR__ .'/error.php');

			return;
		}
	
		RegisterModule($this->MODULE_ID);

		$this->InstallDB();
		$this->InstallFiles();
		$this->InstallEvents();
		$this->InstallAgents();
	}

	/**
	 * Процесс удаления модуля
	 */
	public function DoUninstall()
	{
		$this->UnInstallDB();
		$this->UnInstallFiles();
		$this->UnInstallEvents();
		$this->UnInstallAgents();

		COption::RemoveOption($this->MODULE_ID);
		CAgent::RemoveModuleAgents($this->MODULE_ID);
		UnRegisterModule($this->MODULE_ID);
	}

	/**
	 * Создание структуры БД модуля
	 */
	public function InstallDB()
	{
		return $this->execSqlInDir(__DIR__ .'/db/install/');
	}

	/**
	 * Удаление структуры БД модуля
	 */
	public function UnInstallDB()
	{
		return $this->execSqlInDir(__DIR__ .'/db/uninstall/');
	}

	/**
	 * Копирование файлов
	 */
	public function InstallFiles()
	{
		foreach($this->filesToCopy() as $src => $dest) {
			mkdir(dirname($dest), BX_DIR_PERMISSIONS, true);
			copy($src, $dest);
		}

		return true;
	}

	/**
	 * Удаление ранее скопированных файлов
	 */
	public function UnInstallFiles()
	{
		foreach($this->filesToCopy() as $src => $dest) {
			unlink($dest);
		}

		return true;
	}

	/**
	 * Подписка на события системы модулем
	 */
	public function InstallEvents()
	{
		foreach($this->getEvents() as $arEvent) {
			\Bitrix\Main\EventManager::getInstance()->registerEventHandler(
				$arEvent['module'],
				$arEvent['name'],
				$this->MODULE_ID,
				$arEvent['callback'] ? $arEvent['callback'][0] : '',
				$arEvent['callback'] ? $arEvent['callback'][1] : '',
				$arEvent['sort']     ?: 100,
				$arEvent['path']     ?: '',
				$arEvent['args']     ?: array()
			);
		}

		return true;
	}

	/**
	 * Удаление подписчиков модуля на события системы
	 */
	public function UnInstallEvents()
	{
		foreach ($this->getEvents() as $arEvent) {
			\Bitrix\Main\EventManager::getInstance()->unRegisterEventHandler(
				$arEvent['module'],
				$arEvent['name'],
				$this->MODULE_ID,
				$arEvent['callback'] ? $arEvent['callback'][0] : '',
				$arEvent['callback'] ? $arEvent['callback'][1] : '',
				$arEvent['path']     ?: '',
				$arEvent['args']     ?: array()
			);
		}

		return true;
	}

	/**
	 * Добавление агентов модуля
	 */
	public function InstallAgents()
	{
		foreach($this->getAgents() as $arAgent) {
			CAgent::AddAgent(
				$arAgent['callback'],
				$this->MODULE_ID,
				$arAgent['period']    ?: 'N',
				$arAgent['interval']  ?: 86400,
				$arAgent['datecheck'] ?: '',
				$arAgent['active']    ?: 'Y',
				$arAgent['next_exec'] ?: '',
				$arAgent['sort']      ?: 100
			);
		}
	}

	/**
	 * Удаляет агентов модуля
	 */
	public function UnInstallAgents()
	{
		CAgent::RemoveModuleAgents($this->MODULE_ID);
	}

	/**
	 * Ищет и выполняет все sql файлы в директории
	 * 
	 * @param  string $dir директория для поиска
	 * @return bool        результат выполнения запросов
	 */
	protected function execSqlInDir($dir)
	{
		$dir = rtrim($dir, '/') .'/';
		foreach (glob($dir . '*.sql') as $file) {
			$errors = $GLOBALS['DB']->RunSQLBatch($file);

			if (!empty($errors)) {
				$APPLICATION->ThrowException(implode('', $errors));
				return false;
			}
		}

		return true;
	}

	/**
	 * Возвращает ассоциативный массив содержащий файлы для копирования.
	 * Ключи массива - путь файла источника, 
	 * Значение массива - путь файла назначения. 
	 * Считается что файлы копируются в DOCUMENT_ROOT с сохранением иерархии 
	 * 
	 * @param  boolean $dir     директория для поиска.
	 * @param  string  $pattern маска для поиска.
	 * @return array
	 */
	protected function filesToCopy($dir = false, $pattern = '{,.}*', $level = 0)
	{
		$dir = $dir ?:  __DIR__ ."/files/";
		$dir = rtrim($dir, '/') . '/';

		// ищем все файлы в директории
		$files = array_filter(glob($dir . $pattern, GLOB_BRACE), 'is_file');

		// чиатем вложенные директории
		foreach(glob($dir . '{,.}[!.,!..]*', GLOB_ONLYDIR|GLOB_NOSORT|GLOB_MARK|GLOB_BRACE) as $file) {
			$files = array_merge($files, $this->filesToCopy($file, $pattern, $level + 1));
		}

		// фиксим путь для копирования
		if ($level === 0) {
			$result = array();
			foreach ($files as $file) {
				$result[$file] = str_replace($dir, $_SERVER['DOCUMENT_ROOT'] .'/', $file);
			}

			return $result;
		}

		return $files;
	}

	/**
	 * Загружает список событий модуля из файлов
	 * 
	 * @return array
	 */
	protected function getEvents()
	{
		$events = array();
		foreach (glob(__DIR__ .'/events/*.php') as $file) {
			$events = array_merge($events, include($file));
		}

		return $events;
	}

	/**
	 * Загружает список агентов модуля из файлов
	 * 
	 * @return array
	 */
	protected function getAgents()
	{
		$agents = array();
		foreach (glob(__DIR__ .'/agents/*.php') as $file) {
			$agents = array_merge($agents, include($file));
		}

		return $agents;
	}

	/**
	 * Проверяет зависимости модуля
	 * 
	 * @return string|true
	 */
	protected function checkDependences()
	{
		if (!function_exists('curl_init')) {
			return GetMessage($this->MODULE_ID .'_INSTALL_ERROR_CURL_NOT_FOUND');
		}

		if (!CModule::IncludeModule('sale')) {
			return GetMessage($this->MODULE_ID .'_INSTALL_ERROR_SALE_NOT_FOUND');
		}

		if (!class_exists('\\Bitrix\\Sale\\Delivery\\Services\\Table')) {
			return GetMessage($this->MODULE_ID .'_INSTALL_ERROR_SALE_BAD_VERSION');
		}

		return true;
	}
}